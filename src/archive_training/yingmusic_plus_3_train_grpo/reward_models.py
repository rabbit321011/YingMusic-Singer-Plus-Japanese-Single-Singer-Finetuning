"""
GRPO Reward Models: Whisper Large V3 / WavLM-large / torchcrepe / DNSMOS P.808
所有模型单例加载，提供统一评分接口。
"""

import json
import os
import torch
import torchaudio
import torchcrepe
import numpy as np

from pykakasi import kakasi
from faster_whisper import WhisperModel
from transformers import WavLMModel
from speechmos import dnsmos

# ---------------------------------------------------------------------------
# 路径配置（服务器）
# ---------------------------------------------------------------------------
WAVLM_PATH = "${REMOTE_ROOT}/pretrained_models/wavlm-large"

# torchcrepe 参数
CREPE_SR = 16000
CREPE_HOP = 160
CREPE_FMIN = 50
CREPE_FMAX = 2000

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

_kks = kakasi()


def _to_kana(text: str) -> str:
    """日文 → 平假名（与 pykakasi 新 API 兼容）"""
    return "".join([c["hira"] for c in _kks.convert(text)])


def _cer(ref: str, hyp: str) -> float:
    """Character Error Rate (Levenshtein distance / |ref|)"""
    ref = ref.replace(" ", "").replace("ー", "")
    hyp = hyp.replace(" ", "").replace("ー", "")
    m, n = len(ref), len(hyp)
    if m == 0:
        return 1.0 if n > 0 else 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1),
            )
    return dp[m][n] / m


def _load_audio_16k(path: str) -> torch.Tensor:
    """加载音频重采样到 16kHz mono，返回 [samples] tensor"""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)


# ---------------------------------------------------------------------------
# RewardModels 主类
# ---------------------------------------------------------------------------

class RewardModels:
    """四路 reward 模型封装，单例加载。use_remote_whisper=True 时将 PER 外包给独立 GPU"""

    def __init__(self, device: torch.device, whisper_batch_size: int = 32,
                 use_remote_whisper: bool = False):
        self.device = device
        self._loaded = False
        self._whisper_batch_size = whisper_batch_size
        self._whisper = None
        self._wavlm = None
        self._wavlm_device = None
        self._use_remote = use_remote_whisper
        if self._use_remote:
            RemoteWhisperClient.check_server(host="127.0.0.1", port=15555)

    # ---- 加载 ----

    def load_all(self):
        """加载全部 reward 模型。远程 Whisper 模式跳过本地加载。"""
        if self._loaded:
            return
        if self._use_remote:
            self._remote_whisper = RemoteWhisperClient()
        else:
            self._load_whisper()
        self._load_wavlm()
        self._loaded = True

    def _load_whisper(self):
        print("[Reward] Loading Whisper Large V3 ...", flush=True)
        self._whisper = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="int8_float16",  # int8 推理省显存 (~6GB vs ~10.9GB fp16)
        )
        print("[Reward] Whisper Large V3 ready.", flush=True)

    def _load_wavlm(self):
        print(f"[Reward] Loading WavLM-large from {WAVLM_PATH} ...", flush=True)
        self._wavlm = WavLMModel.from_pretrained(
            WAVLM_PATH, local_files_only=True
        )
        self._wavlm_device = self.device  # 与 Whisper 同 GPU
        self._wavlm = self._wavlm.to(self._wavlm_device).eval()
        print("[Reward] WavLM-large ready.", flush=True)

    # ---- PER: Whisper Large V3 ----

    def score_per_batch(
        self, audio_paths: list[str], ref_kanas: list[str]
    ) -> torch.Tensor:
        """批量计算 PER reward。远程模式下走 TCP。"""
        if self._use_remote:
            return self._remote_whisper.score_per(audio_paths, ref_kanas)

        assert self._whisper is not None, "Whisper not loaded"
        scores = torch.zeros(len(audio_paths))

        # 分批转录（faster-whisper transcribe 不支持 batch，逐条处理）
        for i, path in enumerate(audio_paths):
            try:
                segments, _ = self._whisper.transcribe(
                    path, language="ja", beam_size=5
                )
                hyp_text = " ".join([s.text for s in segments])
                hyp_kana = _to_kana(hyp_text)
                ref_kana = ref_kanas[i]
                cer_val = _cer(ref_kana, hyp_kana)
                scores[i] = max(0.0, 1.0 - cer_val)
            except Exception as e:
                print(f"[Reward PER] Error on {os.path.basename(path)}: {e}", flush=True)
                scores[i] = -1.0  # OOM 标记：由 advantage 计算时用组内均值填充

        return scores

    def free_whisper(self):
        """释放 Whisper 显存（远程模式下无操作）"""
        if self._use_remote:
            return
        if self._whisper is not None:
            del self._whisper
            self._whisper = None
            torch.cuda.empty_cache()

    def ensure_whisper(self):
        """确保 Whisper 已加载（远程模式下无操作）"""
        if self._use_remote:
            return
        if self._whisper is None:
            self._load_whisper()

    # ---- SIM: WavLM-large ----

    @torch.no_grad()
    def score_sim_batch(
        self, gen_wavs: list[torch.Tensor], ref_wavs: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        批量计算音色相似度 reward。
        gen_wavs/ref_wavs: 每个元素是 [samples] 16kHz mono tensor。
        返回 [N] tensor，范围约 [0.85, 0.95]。
        """
        assert self._wavlm is not None, "WavLM not loaded"
        device = self._wavlm_device
        scores = torch.zeros(len(gen_wavs))

        for i, (gw, rw) in enumerate(zip(gen_wavs, ref_wavs)):
            gw = gw.unsqueeze(0).to(device)
            rw = rw.unsqueeze(0).to(device)
            out_g = self._wavlm(gw).last_hidden_state.mean(dim=1)  # [1, D]
            out_r = self._wavlm(rw).last_hidden_state.mean(dim=1)  # [1, D]
            out_g = out_g / out_g.norm(p=2, dim=-1, keepdim=True)
            out_r = out_r / out_r.norm(p=2, dim=-1, keepdim=True)
            scores[i] = float((out_g * out_r).sum())

        return scores

    # ---- F0-CORR: torchcrepe ----

    @torch.no_grad()
    def score_f0_batch(
        self, gen_wavs: list[torch.Tensor], ref_wavs: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        批量计算 F0 Pearson 相关性 reward。
        仅比较 both-voiced 帧。返回 [N] tensor，范围约 [0.9, 1.0]。
        """
        scores = torch.zeros(len(gen_wavs))

        for i, (gw, rw) in enumerate(zip(gen_wavs, ref_wavs)):
            # 对齐长度
            min_len = min(gw.shape[-1], rw.shape[-1])
            gw_aligned = gw[:min_len].unsqueeze(0).to(self.device)
            rw_aligned = rw[:min_len].unsqueeze(0).to(self.device)

            f0_g, p_g = torchcrepe.predict(
                gw_aligned, CREPE_SR, hop_length=CREPE_HOP,
                fmin=CREPE_FMIN, fmax=CREPE_FMAX, model="full",
                batch_size=512, device=self.device, return_periodicity=True,
            )
            f0_r, p_r = torchcrepe.predict(
                rw_aligned, CREPE_SR, hop_length=CREPE_HOP,
                fmin=CREPE_FMIN, fmax=CREPE_FMAX, model="full",
                batch_size=512, device=self.device, return_periodicity=True,
            )

            f0_g = f0_g.squeeze().cpu().numpy()
            f0_r = f0_r.squeeze().cpu().numpy()
            voiced_g = p_g.squeeze().cpu().numpy() > 0.5
            voiced_r = p_r.squeeze().cpu().numpy() > 0.5

            # 对齐帧数
            n_frames = min(len(f0_g), len(f0_r))
            both = voiced_g[:n_frames] & voiced_r[:n_frames]
            if both.sum() >= 10:
                corr = np.corrcoef(f0_g[:n_frames][both], f0_r[:n_frames][both])[0, 1]
                scores[i] = max(0.0, float(corr) if not np.isnan(corr) else 0.0)
            else:
                scores[i] = 0.0

        return scores

    # ---- DNSMOS P.808 ----

    def score_dnsmos_batch(self, audio_paths: list[str]) -> torch.Tensor:
        """
        批量计算 DNSMOS P.808（CPU ONNX）。
        返回 [N] tensor，归一化到 [0, 1]（除以 5.0）。
        """
        scores = torch.zeros(len(audio_paths))
        for i, path in enumerate(audio_paths):
            try:
                result = dnsmos.run(path, sr=16000, return_df=False)
                mos = float(result["p808_mos"])
                scores[i] = mos / 5.0  # 归一化
            except Exception as e:
                print(f"[Reward DNSMOS] Error on {os.path.basename(path)}: {e}", flush=True)
                scores[i] = 0.5  # 中性兜底
        return scores

    # ---- 综合打分 ----

    def score_all(
        self,
        tmp_wav_paths: list[str],
        ref_audio_16k: list[torch.Tensor],
        ref_kanas: list[str],
    ) -> torch.Tensor:
        """三路/四路综合打分，返回 [N, 4] tensor: [PER, SIM, F0, DNSMOS]"""
        # DNSMOS (CPU)
        dns_scores = self.score_dnsmos_batch(tmp_wav_paths)
        # Whisper PER
        self.ensure_whisper()
        per_scores = self.score_per_batch(tmp_wav_paths, ref_kanas)
        # WavLM SIM + torchcrepe F0
        gen_wavs_16k = [_load_audio_16k(p) for p in tmp_wav_paths]
        sim_scores = self.score_sim_batch(gen_wavs_16k, ref_audio_16k)
        f0_scores = self.score_f0_batch(gen_wavs_16k, ref_audio_16k)
        return torch.stack([per_scores, sim_scores, f0_scores, dns_scores], dim=1)


# ---------------------------------------------------------------------------
# RemoteWhisperClient: 轻量客户端，仅做 PER 打分
# ---------------------------------------------------------------------------

class RemoteWhisperClient:
    """将 Whisper PER 打分外包给独立 GPU 上的 reward_server"""

    def __init__(self, host: str = "127.0.0.1", port: int = 15555, timeout: float = 120.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    @staticmethod
    def check_server(host="127.0.0.1", port=15555):
        """启动时检查 server 是否可达"""
        try:
            import time
            for _ in range(60):  # 最多等 60s
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                if s.connect_ex((host, port)) == 0:
                    s.close()
                    return
                s.close()
                time.sleep(1)
        except:
            pass
        print("[Reward] WARNING: Remote Whisper server not responding!", flush=True)

    def score_per(self, audio_paths: list[str], ref_kanas: list[str]) -> torch.Tensor:
        """返回 [N] PER scores"""
        req = json.dumps({"paths": audio_paths, "ref_kanas": ref_kanas}).encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect((self.host, self.port))
            sock.sendall(struct.pack("!I", len(req)) + req)
            header = self._recv_exact(sock, 4)
            size = struct.unpack("!I", header)[0]
            data = self._recv_exact(sock, size)
            return pickle.loads(data)
        finally:
            sock.close()

    @staticmethod
    def _recv_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Remote Whisper server disconnected")
            buf += chunk
        return buf


# ---------------------------------------------------------------------------
# RemoteRewardClient: TCP 客户端，连接 reward_server.py (GPU set)
# ---------------------------------------------------------------------------

import pickle
import socket
import struct


class RemoteRewardClient:
    """替代本地 RewardModels，通过 TCP 请求 GPU set 上的 reward_server 打分。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 15555, timeout: float = 600.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def score_all(
        self,
        gen_wav_paths: list[str],
        ref_wav_paths: list[str],
        ref_kanas: list[str],
    ) -> torch.Tensor:
        """
        发送打分请求到远程 reward server，返回 [N, 4] scores tensor。
        列: [PER, SIM, F0, DNSMOS]
        """
        req = json.dumps({
            "gen_wavs": gen_wav_paths,
            "ref_wavs": ref_wav_paths,
            "ref_kanas": ref_kanas,
        }).encode("utf-8")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect((self.host, self.port))
            # 发送
            sock.sendall(struct.pack("!I", len(req)) + req)
            # 接收
            header = self._recv_exact(sock, 4)
            size = struct.unpack("!I", header)[0]
            data = self._recv_exact(sock, size)
            return pickle.loads(data)
        finally:
            sock.close()

    @staticmethod
    def _recv_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Reward server disconnected")
            buf += chunk
        return buf

