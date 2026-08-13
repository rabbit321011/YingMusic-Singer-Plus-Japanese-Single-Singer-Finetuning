import torch.nn.functional as F


REFERENCE_MODE_SPLIT = "split"
REFERENCE_MODE_ALL_B = "all_b"
REFERENCE_MODES = (REFERENCE_MODE_SPLIT, REFERENCE_MODE_ALL_B)


def resolve_ref_len(reference_mode, split_ref_len, total_frames):
    total_frames = int(total_frames)
    split_ref_len = int(split_ref_len)
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if reference_mode == REFERENCE_MODE_ALL_B:
        return 0
    if reference_mode != REFERENCE_MODE_SPLIT:
        raise ValueError(f"unknown reference mode: {reference_mode}")
    if not 0 < split_ref_len < total_frames:
        raise ValueError("split ref_len must create non-empty A/B regions")
    return split_ref_len


def compute_flow_losses(
    v_pred,
    v_target,
    ref_len,
    flow_b_weight,
    phase,
    reference_mode,
):
    if v_pred.shape != v_target.shape:
        raise ValueError("flow prediction and target shapes differ")
    if v_pred.ndim != 3 or v_pred.shape[1] <= 0:
        raise ValueError("flow tensors must have shape [B, T, D] with T > 0")

    ref_len = int(ref_len)
    total_frames = int(v_pred.shape[1])
    if reference_mode == REFERENCE_MODE_ALL_B:
        if ref_len != 0:
            raise ValueError("all_b mode requires ref_len=0")
        flow_a = v_pred.new_zeros(())
        flow_b = F.mse_loss(v_pred, v_target)
    elif reference_mode == REFERENCE_MODE_SPLIT:
        if not 0 < ref_len < total_frames:
            raise ValueError("split mode requires non-empty A/B regions")
        flow_a = F.mse_loss(v_pred[:, :ref_len, :], v_target[:, :ref_len, :])
        flow_b = F.mse_loss(v_pred[:, ref_len:, :], v_target[:, ref_len:, :])
    else:
        raise ValueError(f"unknown reference mode: {reference_mode}")

    flow = flow_b if phase == "p_only" else flow_a + flow_b_weight * flow_b
    return flow_a, flow_b, flow
