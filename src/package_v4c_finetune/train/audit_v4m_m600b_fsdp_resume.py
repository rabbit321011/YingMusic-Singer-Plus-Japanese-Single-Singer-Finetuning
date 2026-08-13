"""Audit a continuous M600-B FSDP run against a stop-and-resume run."""

from __future__ import annotations

import argparse
import json
import pathlib


REPORT_SCHEMA = "v4m_m600b_fsdp_run_report_v1"
AUDIT_SCHEMA = "v4m_m600b_fsdp_resume_audit_v1"


def load_report(path):
    path = pathlib.Path(path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "ok":
        raise ValueError(f"invalid M600-B FSDP report: {path}")
    return path, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--expected_step", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    continuous_path, continuous = load_report(args.continuous)
    resumed_path, resumed = load_report(args.resumed)
    checks = {
        "contract_exact": continuous["contract"] == resumed["contract"],
        "end_step_exact": (
            continuous["end_step"] == resumed["end_step"] == args.expected_step
        ),
        "rank_count_exact": len(continuous["ranks"]) == len(resumed["ranks"]),
        "all_blocks_changed_continuous": continuous["changed_blocks"]
        == list(range(31)),
        "all_blocks_changed_resumed": resumed["changed_blocks"] == list(range(31)),
    }
    rank_results = []
    if checks["rank_count_exact"]:
        continuous_ranks = {int(item["rank"]): item for item in continuous["ranks"]}
        resumed_ranks = {int(item["rank"]): item for item in resumed["ranks"]}
        checks["rank_ids_exact"] = continuous_ranks.keys() == resumed_ranks.keys()
        if checks["rank_ids_exact"]:
            for rank in sorted(continuous_ranks):
                before = continuous_ranks[rank]
                after = resumed_ranks[rank]
                result = {
                    "rank": rank,
                    "trace_exact": before["trace"] == after["trace"],
                    "fingerprints_exact": (
                        before["fingerprints"] == after["fingerprints"]
                    ),
                    "fingerprints": after["fingerprints"],
                }
                rank_results.append(result)
        else:
            rank_results = []
    checks["all_rank_traces_exact"] = bool(rank_results) and all(
        item["trace_exact"] for item in rank_results
    )
    checks["all_rank_fingerprints_exact"] = bool(rank_results) and all(
        item["fingerprints_exact"] for item in rank_results
    )
    passed = all(checks.values())
    audit = {
        "schema": AUDIT_SCHEMA,
        "passed": passed,
        "continuous": str(continuous_path),
        "resumed": str(resumed_path),
        "expected_step": args.expected_step,
        "checks": checks,
        "ranks": rank_results,
    }
    output = pathlib.Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("M600-B FSDP strict resume audit failed")


if __name__ == "__main__":
    main()
