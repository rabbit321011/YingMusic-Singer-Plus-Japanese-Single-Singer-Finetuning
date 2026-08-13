import json, sys
sys.path.insert(0, '.')
from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer

tokenizer = CNENTokenizer()
with open('${SERVER_ROOT}/final_sum_large/train_singnet.json') as f:
    data = json.load(f)

test_samples = data[:5]
print(f'Total train samples: {len(data)}')
print()
for s in test_samples:
    text = s['Text']
    tokens = tokenizer.encode(text)
    dur = s['Duration']
    frames = int(dur * 21.533)
    short = text[:40] + ('...' if len(text) > 40 else '')
    print(f'  Text: {short}')
    print(f'    Tokens={len(tokens)} | Dur={dur:.1f}s | Frames={frames}')
    print(f'    Token[:10]: {tokens[:10]}')
    print()

errors = 0
for i, s in enumerate(data):
    try:
        tokens = tokenizer.encode(s['Text'])
        if len(tokens) == 0:
            print(f'  WARN[{i}]: empty tokens for: {s["Text"][:40]}')
            errors += 1
    except Exception as e:
        print(f'  ERROR[{i}]: {e} | text: {s["Text"][:40]}')
        errors += 1

print(f'Tokenization check: {len(data)-errors}/{len(data)} OK ({errors} errors)')

















