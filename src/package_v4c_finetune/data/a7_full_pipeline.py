#!/usr/bin/env python3
"""A7 全量对齐处理：train(15122)+test(307) → 4档分拣
用法: python a7_full_pipeline.py
依赖: yysinger conda env (faster_whisper 1.2.1, torch 2.9.1+cu128)
"""

import json, os, sys, time
import difflib
import numpy as np
from tqdm import tqdm
from faster_whisper import WhisperModel

# ============ 配置 ============
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.pop('HF_ENDPOINT', None)
os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')
AUDIO_DIR = '${REMOTE_ROOT}/final_sum_large/audio'
INPUT_DIR = '${REMOTE_ROOT}/final_sum_large/pretreatment_text'
OUTPUT_DIR = '${REMOTE_ROOT}/final_sum_large/pretreatment_text/timeset'
DEVICE = 'cuda'
COMPUTE_TYPE = 'float16'
MODEL_NAME = 'Systran/faster-whisper-large-v3'

# 分档规则
def get_tier(sim, n_phrases):
    if sim >= 0.9 and n_phrases <= 10:
        return 'L1_high'
    elif sim >= 0.7:
        return 'L2_medium'
    elif sim >= 0.5:
        return 'L3_low'
    else:
        return 'L4_discard'

# ============ 主逻辑 ============
def transcribe_with_words(model, audio_path):
    try:
        segments, info = model.transcribe(
            audio_path, word_timestamps=True, language='ja',
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
    except Exception as e:
        return None, str(e)
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                word = w.word.strip()
                if word:
                    words.append({'word': word, 'start': w.start, 'end': w.end, 'prob': w.probability})
    return words, None

def fuzzy_match_phrases(words, phrases):
    asr_text = ''.join([w['word'] for w in words])
    target_text = ''.join([p['text'] for p in phrases])
    sm = difflib.SequenceMatcher(None, asr_text, target_text)

    word_positions = []
    pos = 0
    for w in words:
        word_positions.append(pos)
        pos += len(w['word'])

    aligned = []
    pi = 0
    asr_pos = 0
    tgt_pos = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('equal', 'replace'):
            while pi < len(phrases) and tgt_pos < j2:
                p = phrases[pi]
                p_len = len(p['text'])
                if tgt_pos + p_len <= j2:
                    match_chars = p_len
                    wi_start = -1
                    wi_end = -1
                    for wi, wp in enumerate(word_positions):
                        if wp >= asr_pos and wi_start < 0:
                            wi_start = wi
                        if wi_start >= 0 and wp + len(words[wi]['word']) >= asr_pos + match_chars:
                            wi_end = wi + 1
                            break
                    if wi_end < 0:
                        wi_end = len(words)
                    if wi_end > wi_start >= 0:
                        p_start = words[wi_start]['start']
                        p_end = words[wi_end - 1]['end']
                        aligned.append({
                            'text': p['text'],
                            'kana': p.get('kana', ''),
                            'start': round(p_start, 3),
                            'end': round(p_end, 3),
                            'asr_start': round(p_start, 3),
                            'asr_end': round(p_end, 3),
                            'match_quality': tag
                        })
                    asr_pos += match_chars
                    tgt_pos += p_len
                    pi += 1
                else:
                    break
    return aligned

def process_split(model, data, split_name):
    tiers = {'L1_high': [], 'L2_medium': [], 'L3_low': [], 'L4_discard': []}
    stats = {'total': len(data), 'success': 0, 'vad_skip': 0, 'error': 0,
             'tier_counts': {}, 'total_time': 0}

    pbar = tqdm(data, desc=split_name, unit='samp', ncols=100)
    for item in pbar:
        filename = os.path.basename(item['Path'])
        audio_path = os.path.join(AUDIO_DIR, filename)
        phrases = item['Phrases']
        n_phrases = len(phrases)

        if not os.path.exists(audio_path):
            stats['error'] += 1
            continue

        ta = time.time()
        words, err = transcribe_with_words(model, audio_path)
        asr_time = time.time() - ta
        stats['total_time'] += asr_time

        if err:
            stats['error'] += 1
            continue

        if not words:
            stats['vad_skip'] += 1
            continue

        asr_text = ''.join([w['word'] for w in words])
        target_text = ''.join([p['text'] for p in phrases])
        sim = difflib.SequenceMatcher(None, asr_text, target_text).ratio()
        matched = fuzzy_match_phrases(words, phrases)
        tier = get_tier(sim, n_phrases)

        output = {
            'Path': item['Path'],
            'Duration': item['Duration'],
            'Text': item['Text'],
            'Language': item['Language'],
            'Phrases': matched,
            'ASRSim': round(sim, 4),
            'ASRText': asr_text,
            'ASRWords': len(words),
            'Tier': tier,
            'AlignmentMethod': 'A7_whisper_largev3',
            'AlignmentStatus': 'ok' if tier != 'L4_discard' else 'discard'
        }

        tiers[tier].append(output)
        stats['success'] += 1

        if stats['success'] % 50 == 0:
            elapsed = stats['total_time']
            eta = elapsed / stats['success'] * (len(data) - stats['success'])
            pbar.set_postfix({
                'suc': stats['success'], 'vad': stats['vad_skip'],
                'err': stats['error'], 'eta': f'{eta/60:.0f}min'
            })

    pbar.close()
    return tiers, stats

# ============ 主入口 ============
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 60)
    print('加载 Whisper large-v3 (GPU)...')
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE,
                         local_files_only=True)
    print()

    for split_name in ['train_singnet_kana', 'test_singnet_kana']:
        input_path = os.path.join(INPUT_DIR, f'{split_name}.json')
        if not os.path.exists(input_path):
            print(f'文件不存在: {input_path}')
            continue

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f'{split_name}: {len(data)}条')
        tiers, stats = process_split(model, data, split_name)

        # 写入4档文件
        prefix = split_name.replace('_singnet_kana', '')
        for tier_name, items in tiers.items():
            out_path = os.path.join(OUTPUT_DIR, f'{prefix}_{tier_name}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f'  {out_path}: {len(items)}条')

        # 写入统计
        stats_path = os.path.join(OUTPUT_DIR, f'{prefix}_stats.json')
        stats['tier_counts'] = {k: len(v) for k, v in tiers.items()}
        stats['total_time_min'] = round(stats['total_time'] / 60, 1)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f'  成功: {stats["success"]} | VAD跳过: {stats["vad_skip"]} | 错误: {stats["error"]}')
        print(f'  L1: {len(tiers["L1_high"])} | L2: {len(tiers["L2_medium"])} | L3: {len(tiers["L3_low"])} | L4: {len(tiers["L4_discard"])}')
        print(f'  耗时: {stats["total_time_min"]:.1f}min')
        print()

    print('全部完成!')
