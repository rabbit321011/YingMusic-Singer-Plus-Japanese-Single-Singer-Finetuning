import math
from decimal import Decimal, ROUND_HALF_UP

from .placement import (
    DEFAULT_FRAME_RATE,
    DEFAULT_SEP_TOKEN_ID,
    canonical_sha256,
    solve_monotonic_frames,
)


TIME_SCALE = 1_000_000
FRAME_NUMERATOR = 44100
FRAME_DENOMINATOR = 2048


def _microseconds(value):
    return int(
        (Decimal(str(value)) * TIME_SCALE).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def _relative_frame(seconds, reference_seconds):
    delta_microseconds = _microseconds(seconds) - _microseconds(reference_seconds)
    numerator = delta_microseconds * FRAME_NUMERATOR
    denominator = FRAME_DENOMINATOR * TIME_SCALE
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def _punctuation_target(token_events, event_index, phone_mapping, phrase):
    previous_interval = None
    next_interval = None
    for event in reversed(token_events[:event_index]):
        if event["kind"] != "phone" or event["global_phone_index"] is None:
            continue
        previous_interval = phone_mapping[event["global_phone_index"]]
        if previous_interval is not None:
            break
    for event in token_events[event_index + 1 :]:
        if event["kind"] != "phone" or event["global_phone_index"] is None:
            continue
        next_interval = phone_mapping[event["global_phone_index"]]
        if next_interval is not None:
            break

    if previous_interval is not None and next_interval is not None:
        seconds = (float(previous_interval["end"]) + float(next_interval["start"])) / 2
    elif previous_interval is not None:
        seconds = float(phrase["end"])
    elif next_interval is not None:
        seconds = float(phrase["start"])
    else:
        raise ValueError("punctuation has no neighboring phone interval")
    return int(seconds * DEFAULT_FRAME_RATE)


def build_phrase_candidates(
    phrases,
    frontend,
    phone_match,
    total_frames,
    max_abs_phone_shift=None,
    sep_token_id=DEFAULT_SEP_TOKEN_ID,
):
    total_frames = int(total_frames)
    phone_mapping = phone_match["mapping"]
    candidates = []

    for phrase_index, (phrase, phrase_frontend) in enumerate(
        zip(phrases, frontend["phrases"])
    ):
        locked_tokens = [int(token) for token in phrase["tokens"]]
        candidate_tokens = locked_tokens + [int(sep_token_id)]
        fallback_reason = phrase_frontend.get("fallback_reason")
        phrase_phone_indexes = [
            event["global_phone_index"]
            for event in frontend["phone_events"]
            if event["phrase_index"] == phrase_index
        ]

        if fallback_reason is None and phone_match["status"] == "non_cl_phone_edit":
            fallback_reason = "non_cl_phone_edit"
        if fallback_reason is None and any(
            phone_mapping[index] is None for index in phrase_phone_indexes
        ):
            fallback_reason = "missing_cl"

        target_seconds = []
        event_kinds = []
        if fallback_reason is None:
            try:
                for event_index, event in enumerate(phrase_frontend["token_events"]):
                    if event["kind"] == "phone":
                        interval = phone_mapping[event["global_phone_index"]]
                        target = float(interval["start"])
                    else:
                        punctuation_frame = _punctuation_target(
                            phrase_frontend["token_events"],
                            event_index,
                            phone_mapping,
                            phrase,
                        )
                        target = punctuation_frame / DEFAULT_FRAME_RATE
                    target_seconds.append(target)
                    event_kinds.append(event["kind"])
                target_seconds.append(float(phrase["end"]))
                event_kinds.append("sep")
            except (KeyError, TypeError, ValueError) as error:
                fallback_reason = f"target_build:{type(error).__name__}:{error}"

        placement = None
        if fallback_reason is None:
            first_phone_interval = next(
                (
                    phone_mapping[index]
                    for index in phrase_phone_indexes
                    if phone_mapping[index] is not None
                ),
                None,
            )
            if first_phone_interval is None:
                fallback_reason = "missing_first_phone_interval"
            reference_seconds = (
                float(first_phone_interval["start"])
                if first_phone_interval is not None
                else float(phrase["start"])
            )
            target_frames = [
                _relative_frame(seconds, reference_seconds)
                for seconds in target_seconds
            ]
            first_frame = 0
            control_anchor = int(float(phrase["start"]) * DEFAULT_FRAME_RATE)
            next_start = (
                int(float(phrases[phrase_index + 1]["start"]) * DEFAULT_FRAME_RATE)
                if phrase_index + 1 < len(phrases)
                else total_frames
            )
            lower_frame = 0
            upper_frame = min(total_frames - 1, next_start - 1) - control_anchor
            if target_frames:
                target_frames[0] = first_frame
            try:
                placement = solve_monotonic_frames(
                    target_frames,
                    first_frame=first_frame,
                    lower_frame=lower_frame,
                    upper_frame=upper_frame,
                    priority_mask=[kind == "phone" for kind in event_kinds],
                )
            except ValueError as error:
                fallback_reason = f"unrepresentable:{error}"

        max_phone_shift = None
        if placement is not None:
            phone_shifts = [
                shift
                for shift, kind in zip(placement["signed_shifts"], event_kinds)
                if kind == "phone"
            ]
            max_phone_shift = max((abs(shift) for shift in phone_shifts), default=0)
            if (
                max_abs_phone_shift is not None
                and max_phone_shift > int(max_abs_phone_shift)
            ):
                fallback_reason = "phone_shift_exceeds_limit"

        status = "eligible" if fallback_reason is None else "fallback"
        candidate = {
            "phrase_index": phrase_index,
            "status": status,
            "fallback_reason": fallback_reason,
            "tokens": candidate_tokens,
            "token_sha256": canonical_sha256(candidate_tokens),
            "relative_frames": placement["frames"] if placement is not None else [],
            "relative_target_frames": placement["target_frames"] if placement is not None else [],
            "frames": (
                [
                    int(float(phrase["start"]) * DEFAULT_FRAME_RATE) + frame
                    for frame in placement["frames"]
                ]
                if placement is not None
                else []
            ),
            "target_frames": (
                [
                    int(float(phrase["start"]) * DEFAULT_FRAME_RATE) + frame
                    for frame in placement["target_frames"]
                ]
                if placement is not None
                else []
            ),
            "signed_shifts": placement["signed_shifts"] if placement is not None else [],
            "max_abs_shift": placement["max_abs_shift"] if placement is not None else None,
            "max_abs_phone_shift": max_phone_shift,
            "total_abs_shift": placement["total_abs_shift"] if placement is not None else None,
            "collision_count": placement["collision_count"] if placement is not None else None,
        }
        candidates.append(candidate)

    return candidates


def build_fallback_candidates(
    phrases,
    fallback_reason,
    sep_token_id=DEFAULT_SEP_TOKEN_ID,
):
    return [
        {
            "phrase_index": phrase_index,
            "status": "fallback",
            "fallback_reason": str(fallback_reason),
            "tokens": [int(token) for token in phrase["tokens"]]
            + [int(sep_token_id)],
            "token_sha256": canonical_sha256(
                [int(token) for token in phrase["tokens"]] + [int(sep_token_id)]
            ),
            "relative_frames": [],
            "relative_target_frames": [],
            "frames": [],
            "target_frames": [],
            "signed_shifts": [],
            "max_abs_shift": None,
            "max_abs_phone_shift": None,
            "total_abs_shift": None,
            "collision_count": None,
        }
        for phrase_index, phrase in enumerate(phrases)
    ]


def expected_latent_frames(duration):
    return int(math.floor(float(duration) * DEFAULT_FRAME_RATE)) + 1
