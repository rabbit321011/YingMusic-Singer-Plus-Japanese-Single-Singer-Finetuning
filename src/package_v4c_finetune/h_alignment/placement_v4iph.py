import hashlib
import json
import math


DEFAULT_FRAME_RATE = 44100 / 2048
DEFAULT_SEP_TOKEN_ID = 365
DEFAULT_PUL_TOKEN_ID = 366


def canonical_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strictly_increasing(values):
    return all(left < right for left, right in zip(values, values[1:]))


def _is_feasible(
    targets,
    first_frame,
    lower_frame,
    upper_frame,
    max_shift,
    priority_mask,
):
    if not lower_frame <= first_frame <= upper_frame:
        return False
    if priority_mask[0] and abs(first_frame - targets[0]) > max_shift:
        return False

    previous = first_frame
    for target, is_priority in zip(targets[1:], priority_mask[1:]):
        target_lower = target - max_shift if is_priority else lower_frame
        target_upper = target + max_shift if is_priority else upper_frame
        frame = max(previous + 1, lower_frame, target_lower)
        if frame > min(upper_frame, target_upper):
            return False
        previous = frame
    return True


def solve_monotonic_frames(
    target_frames,
    first_frame,
    lower_frame,
    upper_frame,
    priority_mask=None,
):
    """Minimize maximum shift, then total L1 shift, with deterministic ties."""
    targets = [int(value) for value in target_frames]
    first_frame = int(first_frame)
    lower_frame = int(lower_frame)
    upper_frame = int(upper_frame)
    if priority_mask is None:
        priority_mask = [True] * len(targets)
    else:
        priority_mask = [bool(value) for value in priority_mask]

    if not targets:
        raise ValueError("target_frames must not be empty")
    if len(priority_mask) != len(targets):
        raise ValueError("priority_mask length mismatch")
    if lower_frame > upper_frame:
        raise ValueError("lower_frame exceeds upper_frame")
    if len(targets) > upper_frame - lower_frame + 1:
        raise ValueError("not enough frames for strict token order")
    if not lower_frame <= first_frame <= upper_frame:
        raise ValueError("first_frame is outside the allowed range")

    targets[0] = first_frame
    low = 0
    high = max(
        upper_frame - lower_frame,
        max(abs(target - first_frame) for target in targets),
    )
    while low < high:
        middle = (low + high) // 2
        if _is_feasible(
            targets,
            first_frame,
            lower_frame,
            upper_frame,
            middle,
            priority_mask,
        ):
            high = middle
        else:
            low = middle + 1
    max_shift = low
    if not _is_feasible(
        targets,
        first_frame,
        lower_frame,
        upper_frame,
        max_shift,
        priority_mask,
    ):
        raise ValueError("no monotonic placement exists")

    width = upper_frame - lower_frame + 1
    infinity = math.inf
    previous_cost = [infinity] * width
    previous_cost[first_frame - lower_frame] = 0
    backpointers = []

    for token_index in range(1, len(targets)):
        target = targets[token_index]
        current_cost = [infinity] * width
        current_backpointer = [-1] * width

        best_cost = infinity
        best_frame_index = -1
        prefix_cost = [infinity] * width
        prefix_frame_index = [-1] * width
        for frame_index, cost in enumerate(previous_cost):
            if cost < best_cost:
                best_cost = cost
                best_frame_index = frame_index
            prefix_cost[frame_index] = best_cost
            prefix_frame_index[frame_index] = best_frame_index

        if priority_mask[token_index]:
            target_lower = target - max_shift
            target_upper = target + max_shift
        else:
            target_lower = lower_frame
            target_upper = upper_frame
        earliest = max(lower_frame + token_index, target_lower)
        latest = min(
            upper_frame - (len(targets) - token_index - 1),
            target_upper,
        )
        for frame in range(earliest, latest + 1):
            frame_index = frame - lower_frame
            predecessor_index = frame_index - 1
            if predecessor_index < 0:
                continue
            predecessor_cost = prefix_cost[predecessor_index]
            if predecessor_cost == infinity:
                continue
            current_cost[frame_index] = predecessor_cost + abs(frame - target)
            current_backpointer[frame_index] = prefix_frame_index[predecessor_index]

        if all(cost == infinity for cost in current_cost):
            raise ValueError("dynamic program found no placement")
        previous_cost = current_cost
        backpointers.append(current_backpointer)

    if len(targets) == 1:
        frames = [first_frame]
    else:
        final_index = min(
            (index for index, cost in enumerate(previous_cost) if cost != infinity),
            key=lambda index: (previous_cost[index], index),
        )
        reversed_frames = [final_index + lower_frame]
        for pointer_row in reversed(backpointers):
            final_index = pointer_row[final_index]
            if final_index < 0:
                raise AssertionError("broken placement backpointer")
            reversed_frames.append(final_index + lower_frame)
        frames = list(reversed(reversed_frames))

    if frames[0] != first_frame:
        raise AssertionError("first token anchor changed")
    if not _strictly_increasing(frames):
        raise AssertionError("placement is not strictly increasing")
    if any(not lower_frame <= frame <= upper_frame for frame in frames):
        raise AssertionError("placement escaped its allowed range")

    shifts = [frame - target for frame, target in zip(frames, targets)]
    priority_shifts = [
        shift for shift, is_priority in zip(shifts, priority_mask) if is_priority
    ]
    return {
        "target_frames": targets,
        "frames": frames,
        "signed_shifts": shifts,
        "max_abs_shift": max(abs(shift) for shift in shifts),
        "max_abs_priority_shift": max(
            (abs(shift) for shift in priority_shifts),
            default=0,
        ),
        "total_abs_shift": sum(abs(shift) for shift in shifts),
        "collision_count": sum(
            left == right for left, right in zip(targets, targets[1:])
        ),
    }


def _control_phrase_plan(phrase, ref_len, total_frames, sep_token_id):
    phrase_tokens = [int(token) for token in phrase["tokens"]] + [int(sep_token_id)]
    start_frame = int(float(phrase["start"]) * DEFAULT_FRAME_RATE)
    center_frame = (
        (float(phrase["start"]) + float(phrase["end"]))
        * 0.5
        * DEFAULT_FRAME_RATE
    )

    if center_frame < ref_len:
        side = "A"
        anchor = min(max(start_frame, 0), ref_len - 1)
        upper = ref_len - 1
    else:
        side = "B"
        anchor = max(min(start_frame, total_frames - 1), ref_len)
        upper = total_frames - 1

    frames = [
        anchor + index
        for index in range(len(phrase_tokens))
        if anchor + index <= upper
    ]
    return {
        "tokens": phrase_tokens[: len(frames)],
        "frames": frames,
        "anchor": anchor,
        "side": side,
        "side_lower": 0 if side == "A" else ref_len,
        "side_upper": upper,
        "truncated_tokens": len(phrase_tokens) - len(frames),
    }


def _render(
    phrases,
    candidates,
    ref_len,
    total_frames,
    mode,
    sep_token_id,
    force_all_sentence=False,
):
    if not 0 <= ref_len < total_frames:
        raise ValueError("ref_len must leave a non-empty generation region")
    if len(phrases) != len(candidates):
        raise ValueError("phrase/candidate count mismatch")

    text = [0] * total_frames
    occupied = set()
    collision_count = 0
    audits = []
    for phrase_index, (phrase, candidate) in enumerate(zip(phrases, candidates)):
        control = _control_phrase_plan(phrase, ref_len, total_frames, sep_token_id)
        tokens = control["tokens"]
        frames = control["frames"]
        placement_mode = "sentence"
        fallback_reason = None

        if mode == "phone" and force_all_sentence:
            fallback_reason = "sample_control_anomaly"
        elif mode == "phone":
            candidate_tokens = [int(token) for token in candidate.get("tokens", [])]
            if candidate.get("relative_frames"):
                candidate_frames = [
                    control["anchor"] + int(frame)
                    for frame in candidate["relative_frames"]
                ]
            else:
                candidate_frames = [int(frame) for frame in candidate.get("frames", [])]
            if candidate.get("status") != "eligible":
                fallback_reason = candidate.get("fallback_reason") or "offline_ineligible"
            elif control["truncated_tokens"]:
                fallback_reason = "control_truncation"
            elif candidate_tokens != tokens:
                fallback_reason = "runtime_token_mismatch"
            elif len(candidate_frames) != len(tokens):
                fallback_reason = "runtime_frame_count_mismatch"
            elif not candidate_frames or candidate_frames[0] != control["anchor"]:
                fallback_reason = "runtime_first_anchor_mismatch"
            elif not _strictly_increasing(candidate_frames):
                fallback_reason = "runtime_non_monotonic"
            elif any(
                not control["side_lower"] <= frame <= control["side_upper"]
                for frame in candidate_frames
            ):
                fallback_reason = "runtime_ab_boundary"
            elif any(frame in occupied for frame in candidate_frames):
                fallback_reason = "runtime_cross_phrase_collision"
            else:
                frames = candidate_frames
                placement_mode = "phone"

        for token, frame in zip(tokens, frames):
            if token <= 0:
                raise ValueError("locked token IDs must be non-PAD")
            if frame in occupied:
                collision_count += 1
            text[frame] = token
            occupied.add(frame)

        audits.append(
            {
                "phrase_index": phrase_index,
                "placement_mode": placement_mode,
                "fallback_reason": fallback_reason,
                "first_token_frame": frames[0] if frames else None,
                "control_first_token_frame": control["anchor"],
                "token_count": len(tokens),
                "truncated_tokens": control["truncated_tokens"],
            }
        )
    return {
        "text": text,
        "phrases": audits,
        "collision_count": collision_count,
    }


def render_paired_placements(
    phrases,
    candidates,
    ref_len,
    total_frames,
    sep_token_id=DEFAULT_SEP_TOKEN_ID,
):
    control = _render(
        phrases,
        candidates,
        int(ref_len),
        int(total_frames),
        "control",
        int(sep_token_id),
    )
    force_all_sentence = control["collision_count"] > 0 or any(
        phrase["truncated_tokens"] > 0 for phrase in control["phrases"]
    )
    phone = _render(
        phrases,
        candidates,
        int(ref_len),
        int(total_frames),
        "phone",
        int(sep_token_id),
        force_all_sentence=force_all_sentence,
    )

    control_tokens = [token for token in control["text"] if token]
    phone_tokens = [token for token in phone["text"] if token]
    first_frames_match = all(
        control_phrase["first_token_frame"] == phone_phrase["first_token_frame"]
        for control_phrase, phone_phrase in zip(control["phrases"], phone["phrases"])
    )
    if control_tokens != phone_tokens or not first_frames_match:
        phone = _render(
            phrases,
            candidates,
            int(ref_len),
            int(total_frames),
            "phone",
            int(sep_token_id),
            force_all_sentence=True,
        )
        phone_tokens = [token for token in phone["text"] if token]
    if control_tokens != phone_tokens:
        raise AssertionError("Control/H non-PAD token sequence changed")
    for control_phrase, phone_phrase in zip(control["phrases"], phone["phrases"]):
        if control_phrase["first_token_frame"] != phone_phrase["first_token_frame"]:
            raise AssertionError("Control/H first token frame changed")

    token_hash = canonical_sha256(control_tokens)
    return {
        "control": control,
        "phone": phone,
        "token_sha256": token_hash,
        "phone_phrase_count": sum(
            phrase["placement_mode"] == "phone" for phrase in phone["phrases"]
        ),
        "fallback_phrase_count": sum(
            phrase["placement_mode"] != "phone" for phrase in phone["phrases"]
        ),
        "sample_control_anomaly": force_all_sentence,
    }


def _h_pul_exact_control_result(
    phrases,
    candidates,
    ref_len,
    total_frames,
    sep_token_id,
    control,
    reason,
    sample_control_anomaly,
):
    exact = _render(
        phrases,
        candidates,
        ref_len,
        total_frames,
        "phone",
        sep_token_id,
        force_all_sentence=True,
    )
    for audit in exact["phrases"]:
        audit.update(
            {
                "placement_mode": "sentence",
                "fallback_reason": reason,
                "pul_frame_count": 0,
                "sep_frame": None,
            }
        )
    return {
        "control": control,
        "phone_pul": exact,
        "locked_event_token_sha256": canonical_sha256(
            [
                token
                for phrase in phrases
                for token in [
                    *[int(value) for value in phrase["tokens"]],
                    int(sep_token_id),
                ]
            ]
        ),
        "phone_phrase_count": 0,
        "pul_phrase_count": 0,
        "exact_control_phrase_count": len(phrases),
        "pul_frame_count": 0,
        "sample_control_anomaly": bool(sample_control_anomaly),
        "sample_structural_fallback": not bool(sample_control_anomaly),
        "structural_fallback_reason": reason,
    }


def render_h_pul_placements(
    phrases,
    candidates,
    ref_len,
    total_frames,
    sep_token_id=DEFAULT_SEP_TOKEN_ID,
    pul_token_id=DEFAULT_PUL_TOKEN_ID,
):
    """Render H phone timing with repeated PUL spans for runtime fallbacks."""
    ref_len = int(ref_len)
    total_frames = int(total_frames)
    sep_token_id = int(sep_token_id)
    pul_token_id = int(pul_token_id)
    if pul_token_id <= 0 or pul_token_id == sep_token_id:
        raise ValueError("PUL token ID must be positive and distinct from SEP")
    if len(phrases) != len(candidates):
        raise ValueError("phrase/candidate count mismatch")

    control = _render(
        phrases,
        candidates,
        ref_len,
        total_frames,
        "control",
        sep_token_id,
    )
    control_anomaly = control["collision_count"] > 0 or any(
        phrase["truncated_tokens"] > 0 for phrase in control["phrases"]
    )
    if control_anomaly:
        return _h_pul_exact_control_result(
            phrases,
            candidates,
            ref_len,
            total_frames,
            sep_token_id,
            control,
            "sample_control_anomaly",
            True,
        )
    if not phrases:
        return {
            "control": control,
            "phone_pul": control,
            "locked_event_token_sha256": canonical_sha256([]),
            "phone_phrase_count": 0,
            "pul_phrase_count": 0,
            "exact_control_phrase_count": 0,
            "pul_frame_count": 0,
            "sample_control_anomaly": False,
            "sample_structural_fallback": False,
            "structural_fallback_reason": None,
        }

    control_plans = [
        _control_phrase_plan(phrase, ref_len, total_frames, sep_token_id)
        for phrase in phrases
    ]
    anchors = [plan["anchor"] for plan in control_plans]
    sep_frames = [*[(anchor - 1) for anchor in anchors[1:]], total_frames - 1]

    plans = []
    occupied = set()
    for phrase_index, (phrase, candidate, control_plan, sep_frame) in enumerate(
        zip(phrases, candidates, control_plans, sep_frames)
    ):
        anchor = int(control_plan["anchor"])
        lyric_tokens = [int(token) for token in phrase["tokens"]]
        if not lyric_tokens:
            return _h_pul_exact_control_result(
                phrases,
                candidates,
                ref_len,
                total_frames,
                sep_token_id,
                control,
                f"phrase_{phrase_index}:empty_lyric_tokens",
                False,
            )
        if any(
            token <= 0 or token in (sep_token_id, pul_token_id)
            for token in lyric_tokens
        ):
            raise ValueError(f"phrase {phrase_index} contains a reserved token ID")
        if not 0 <= anchor < total_frames or not 0 <= sep_frame < total_frames:
            raise AssertionError("runtime H-PUL boundary escaped the latent timeline")

        expected_candidate_tokens = lyric_tokens + [sep_token_id]
        candidate_tokens = [int(token) for token in candidate.get("tokens", [])]
        if candidate.get("relative_frames"):
            candidate_frames = [
                anchor + int(frame) for frame in candidate["relative_frames"]
            ]
        else:
            candidate_frames = [int(frame) for frame in candidate.get("frames", [])]

        phone_reason = None
        if candidate.get("status") != "eligible":
            phone_reason = candidate.get("fallback_reason") or "offline_ineligible"
        elif candidate_tokens != expected_candidate_tokens:
            phone_reason = "runtime_token_mismatch"
        elif len(candidate_frames) != len(expected_candidate_tokens):
            phone_reason = "runtime_frame_count_mismatch"
        else:
            lyric_frames = candidate_frames[:-1]
            if not lyric_frames or lyric_frames[0] != anchor:
                phone_reason = "runtime_first_anchor_mismatch"
            elif not _strictly_increasing(lyric_frames):
                phone_reason = "runtime_non_monotonic"
            elif any(not anchor <= frame < sep_frame for frame in lyric_frames):
                phone_reason = "runtime_segment_boundary"
            elif any(frame in occupied for frame in [*lyric_frames, sep_frame]):
                phone_reason = "runtime_cross_phrase_collision"

        if phone_reason is None:
            tokens = expected_candidate_tokens
            frames = [*candidate_frames[:-1], sep_frame]
            placement_mode = "phone"
            pul_frame_count = 0
        else:
            pul_start = anchor + len(lyric_tokens)
            pul_frame_count = sep_frame - pul_start
            if pul_frame_count < 1:
                return _h_pul_exact_control_result(
                    phrases,
                    candidates,
                    ref_len,
                    total_frames,
                    sep_token_id,
                    control,
                    f"phrase_{phrase_index}:insufficient_pul_capacity",
                    False,
                )
            frames = list(range(anchor, sep_frame + 1))
            tokens = [
                *lyric_tokens,
                *([pul_token_id] * pul_frame_count),
                sep_token_id,
            ]
            placement_mode = "pul"
            if any(frame in occupied for frame in frames):
                return _h_pul_exact_control_result(
                    phrases,
                    candidates,
                    ref_len,
                    total_frames,
                    sep_token_id,
                    control,
                    f"phrase_{phrase_index}:pul_cross_phrase_collision",
                    False,
                )

        if len(tokens) != len(frames) or not _strictly_increasing(frames):
            raise AssertionError("invalid H-PUL token/frame plan")
        occupied.update(frames)
        plans.append(
            {
                "tokens": tokens,
                "frames": frames,
                "audit": {
                    "phrase_index": phrase_index,
                    "placement_mode": placement_mode,
                    "fallback_reason": phone_reason,
                    "first_token_frame": frames[0],
                    "control_first_token_frame": anchor,
                    "lyric_token_count": len(lyric_tokens),
                    "token_count": len(tokens),
                    "pul_frame_count": pul_frame_count,
                    "sep_frame": sep_frame,
                    "truncated_tokens": 0,
                },
            }
        )

    text = [0] * total_frames
    for plan in plans:
        for token, frame in zip(plan["tokens"], plan["frames"]):
            if text[frame] != 0:
                raise AssertionError("H-PUL renderer overwrote a populated frame")
            text[frame] = token

    expected_lyrics = [
        int(token) for phrase in phrases for token in phrase["tokens"]
    ]
    rendered_lyrics = [
        token
        for token in text
        if token not in (0, sep_token_id, pul_token_id)
    ]
    if rendered_lyrics != expected_lyrics:
        raise AssertionError("H-PUL changed the locked lyric token sequence")
    if sum(token == sep_token_id for token in text) != len(phrases):
        raise AssertionError("H-PUL did not render exactly one SEP per phrase")
    if any(
        plan["audit"]["first_token_frame"]
        != plan["audit"]["control_first_token_frame"]
        for plan in plans
    ):
        raise AssertionError("H-PUL changed a Control first-token anchor")

    phone_pul = {
        "text": text,
        "phrases": [plan["audit"] for plan in plans],
        "collision_count": 0,
    }
    locked_events = [
        token
        for phrase in phrases
        for token in [
            *[int(value) for value in phrase["tokens"]],
            sep_token_id,
        ]
    ]
    return {
        "control": control,
        "phone_pul": phone_pul,
        "locked_event_token_sha256": canonical_sha256(locked_events),
        "phone_phrase_count": sum(
            plan["audit"]["placement_mode"] == "phone" for plan in plans
        ),
        "pul_phrase_count": sum(
            plan["audit"]["placement_mode"] == "pul" for plan in plans
        ),
        "exact_control_phrase_count": 0,
        "pul_frame_count": sum(
            plan["audit"]["pul_frame_count"] for plan in plans
        ),
        "sample_control_anomaly": False,
        "sample_structural_fallback": False,
        "structural_fallback_reason": None,
    }
