"""
Whisper 转录服务器 — 运行在独立 GPU 上，接收路径列表，返回 PER scores。

用法:
  CUDA_VISIBLE_DEVICES=3 python reward_server.py --port 15555
"""
import argparse, json, pickle, socket, struct, sys
import torch

from faster_whisper import WhisperModel
from pykakasi import kakasi

_kks = kakasi()

def _to_kana(text: str) -> str:
    return "".join([c["hira"] for c in _kks.convert(text)])

def _cer(ref: str, hyp: str) -> float:
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


def serve(port=15555, device="cuda"):
    print(f"[Server] Loading Whisper Large V3 on {device}...", flush=True)
    whisper = WhisperModel("large-v3", device=device, compute_type="int8_float16")
    print("[Server] Whisper ready. Listening...", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(5)

    while True:
        conn, addr = sock.accept()
        try:
            header = _recv_exact(conn, 4)
            size = struct.unpack("!I", header)[0]
            data = _recv_exact(conn, size)
            req = json.loads(data.decode("utf-8"))
            paths = req["paths"]
            ref_kanas = req["ref_kanas"]

            scores = torch.zeros(len(paths))
            for i, path in enumerate(paths):
                try:
                    segments, _ = whisper.transcribe(path, language="ja", beam_size=5)
                    hyp = _to_kana(" ".join([s.text for s in segments]))
                    cer = _cer(ref_kanas[i], hyp)
                    scores[i] = max(0.0, 1.0 - cer)
                except Exception as e:
                    print(f"[Server] Error on {path}: {e}", flush=True)
                    scores[i] = -1.0

            resp = pickle.dumps(scores)
            conn.sendall(struct.pack("!I", len(resp)) + resp)
        except Exception as e:
            print(f"[Server] Error: {e}", flush=True)
        finally:
            conn.close()


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf += chunk
    return buf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=15555)
    args = parser.parse_args()
    serve(port=args.port)  # device="cuda" uses GPU set of the visible devices (GPU set)

