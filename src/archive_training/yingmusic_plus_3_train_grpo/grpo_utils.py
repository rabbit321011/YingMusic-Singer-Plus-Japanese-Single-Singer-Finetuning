"""
GRPO 工具函数：Advantage 组内标准化、KL 散度、PPO ratio。
基于 Flow Matching SDE 的高斯概率模型。
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 组内标准化 → Advantage
# ---------------------------------------------------------------------------

def compute_advantages(
    scores: torch.Tensor,  # [G, M] 一个 prompt 组内的 M 维 reward
    weights: torch.Tensor,  # [M] 权重
    eps: float = 1e-8,
) -> torch.Tensor:  # [G]
    """
    GRPO 组内标准化（无需 Value Network）。

    对每维 reward 独立做 z-score 标准化后加权求和：
        A_i = Σ_k w_k * (R_{i,k} - mean(R_{:,k})) / (std(R_{:,k}) + ε)

    -1 标记为 OOM 缺失值，用该维度的非 OOM 组内均值填充，不参与竞争也不被惩罚。
    """
    G, M = scores.shape
    weights = weights.to(scores.device)
    z_scores = torch.zeros(G, M, device=scores.device)

    for k in range(M):
        col = scores[:, k]  # [G]
        mask = col >= 0.0  # True = 有效, False = OOM(-1)
        n_valid = mask.sum().item()

        if n_valid == 0:
            # 全 OOM，该维度全部置 0
            continue
        elif n_valid < G:
            # 部分 OOM：用非 OOM 均值填充
            valid_mean = col[mask].mean()
            col = col.clone()
            col[~mask] = valid_mean
            mu = valid_mean
        else:
            mu = col.mean()

        sigma = col.std() + eps
        z_scores[:, k] = (col - mu) / sigma

    advantages = z_scores @ weights
    return advantages


# ---------------------------------------------------------------------------
# Flow Matching + SDE 概率模型
#
# SDE 过渡: x_{t+dt} = x_t + v_θ(x_t, t) * dt + α * ε * √dt
# 等效 velocity: v_eff = (x_{t+dt} - x_t) / dt ~ N(v_θ, (α²/dt) * I)
#
# log π(v_eff | x_t, θ) ∝ -dt * ||v_eff - v_θ||² / (2 * α²)
# （常数项在 ratio 计算中抵消）
# ---------------------------------------------------------------------------

def _sde_logp_factor(
    v_eff: torch.Tensor,  # [B, T, D] effective velocity
    v_pred: torch.Tensor,  # [B, T, D] predicted velocity
    dt: float,
    alpha_sq: float,
) -> torch.Tensor:  # [B, T] per-frame log prob
    """
    计算 SDE 过渡的对数概率（忽略常数项）。

    返回每帧的 log prob（正比于 -||v_eff - v_pred||²）。
    """
    diff_sq = ((v_eff - v_pred) ** 2).sum(dim=-1)  # [B, T]
    logp = -dt * diff_sq / (2.0 * alpha_sq)
    return logp


def compute_ppo_ratio(
    v_eff: torch.Tensor,    # 实际发生的 velocity
    v_new: torch.Tensor,    # 当前策略预测
    v_old: torch.Tensor,    # 旧策略预测（采样时的策略）
    dt: float,
    alpha_sq: float,
    eps_l: float = 0.002,
    eps_u: float = 0.01,
) -> torch.Tensor:  # [B, T]
    """
    计算 PPO-clip 损失（per frame, per batch element）。

    返回 clip 后的项，不取平均。
    """
    logp_new = _sde_logp_factor(v_eff, v_new, dt, alpha_sq)  # [B, T]
    logp_old = _sde_logp_factor(v_eff, v_old, dt, alpha_sq)  # [B, T]
    ratio = torch.exp(logp_new - logp_old)  # [B, T]
    # 不对称 clip：下界 (1-eps_l), 上界 (1+eps_u)
    ratio_clipped = torch.clamp(ratio, 1.0 - eps_l, 1.0 + eps_u)
    return ratio, ratio_clipped


def compute_kl_divergence(
    v_policy: torch.Tensor,  # 当前策略
    v_ref: torch.Tensor,     # 参考模型
    dt: float,
    alpha_sq: float,
) -> torch.Tensor:  # [B, T]
    """
    两个高斯 N(v_policy, σ²I) 和 N(v_ref, σ²I) 的 KL 散度。

    D_KL = dt * ||v_policy - v_ref||² / (2 * α²)
    返回每帧的 KL 值。
    """
    diff_sq = ((v_policy - v_ref) ** 2).sum(dim=-1)  # [B, T]
    kl = dt * diff_sq / (2.0 * alpha_sq)
    return kl


# ---------------------------------------------------------------------------
# SDE 采样
# ---------------------------------------------------------------------------

def build_sde_timesteps(
    nfe: int,
    t_shift: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    构建带 t_shift 的去噪时间步调度。
    参数
    ----
    nfe: 去噪步数（训练 10，推理 32）
    t_shift: ZipVoice 风格时间步偏移
    """
    t = torch.linspace(0.0, 1.0, nfe + 1, device=device, dtype=dtype)
    # 与训练/推理一致：t_shift * t / (1 - t_shift * t)
    t = t_shift * t / (1.0 + (t_shift - 1.0) * t)
    return t  # [nfe+1]


@torch.no_grad()
def sde_sample(
    dit,                 # DiT 模型（无 CFG）
    x0: torch.Tensor,    # 初始噪声 [1, T, D]
    cond: torch.Tensor,  # condition latent [1, T, D]
    text: torch.Tensor,  # text tokens [1, T]
    midi: torch.Tensor,  # midi features [1, T, 128]
    t_schedule: torch.Tensor,  # 时间步 [nfe+1]
    sde_window_start: int = 1,
    sde_window_end: int = 9,
    alpha: float = 0.8,
):
    """
    SDE 采样：在指定窗口内注入噪声，其余步走确定性 ODE。

    返回
    ----
    x_t: 最终 latent [1, T, D]
    sde_trajectories: list of dict，每项 = {t_idx, x_t, x_next, v_pred, dt}
                      仅包含 SDE 窗口内的步
    """
    nfe = len(t_schedule) - 1  # 10
    x_t = x0.clone()
    sde_trajectories = []

    for i in range(nfe):
        t_now = t_schedule[i]       # 标量 tensor
        t_next = t_schedule[i + 1]
        dt = float(t_next - t_now)

        # 预测速度场 v_θ(x_t, t_now)
        v_pred, _ = dit(
            x=x_t,
            cond=cond,
            text=text,
            time=t_now,
            midi=midi,
            drop_audio_cond=False,
            drop_text=False,
            drop_midi=False,
            cfg_infer=False,
            cache=False,
        )  # [1, T, D]

        # Euler 步进
        x_next = x_t + v_pred * dt

        # SDE 窗口内注入噪声
        in_sde = (sde_window_start <= i < sde_window_end)
        if in_sde:
            noise = torch.randn_like(x_t)
            x_next = x_next + alpha * noise * (dt ** 0.5)

        # 保存 SDE 窗内的轨迹（用于后续 policy gradient）
        if in_sde:
            sde_trajectories.append({
                "t_idx": i,
                "t_now": t_now,
                "t_next": t_next,
                "dt": dt,
                "x_t": x_t.clone(),
                "x_next": x_next.clone(),
                "v_pred": v_pred.clone(),
            })

        x_t = x_next

    return x_t, sde_trajectories


# ---------------------------------------------------------------------------
# AB 区准备：计算 ref_len（随机短语边界）
# ---------------------------------------------------------------------------

FRAME_RATE = 44100 / 2048  # ~21.53 Hz


def compute_ref_len_grpo(
    T: int,
    phrase_boundaries: list[int],
) -> int:
    """
    GRPO 版 ref_len：随机选一个中间短语边界。

    约束：A ≠ ∅, B ≠ ∅（选 phrase_boundaries[1:-1]）
    """
    if len(phrase_boundaries) >= 3:
        # 有至少 3 个边界 → 随机选中间一个
        import random
        return int(random.choice(phrase_boundaries[1:-1]))
    elif len(phrase_boundaries) == 2:
        # 只有 2 个边界，用第一个（1 个短语在 A，其余在 B）
        return int(phrase_boundaries[1])
    else:
        # 无边界信息 → 随机 12.5%-33%（fallback）
        import random
        ref_len = T * random.uniform(0.125, 0.33)
        ref_len = max(ref_len, int(5.0 * FRAME_RATE))  # 至少 5 秒
        ref_len = min(ref_len, int(T * 0.65))  # 最多 65%
        return int(ref_len)

