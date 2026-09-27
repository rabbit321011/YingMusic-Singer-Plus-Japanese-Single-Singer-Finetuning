"""Create reviewable, sanitized copies of selected research documents.

Run this against a private source tree, then audit the generated diff before
committing. It deliberately excludes source snapshots, media, manuscripts,
archives, and operational/cloud-storage notes.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


SELECTED = (
    "ONBOARDING.md",
    "基础设施/v3/training-plan.md",
    "基础设施/v4/训练/",
    "基础设施/v5/GRPO/GRPO完整计划.md",
    "基础设施/v5/V/",
    "基础设施/v5/V4Pvf/",
    "基础设施/v5/V5-额外压缩通道实验/",
    "基础设施/v5/V5P/",
    "基础设施/数据管线/DATA_PIPELINE.md",
    "基础设施/架构/",
    "基础设施/部署运维/DEPLOY.md",
    "基础设施/部署运维/MSST_linux_setup.md",
    "基础设施/部署运维/MSST_models.md",
    "基础设施/部署运维/SERVER_TRAIN.md",
    "基础设施/归档/main_归档.md",
    "基础设施/归档/a7-mfa-experiment-handoff-20260712.md",
    "基础设施/归档/gtsinger_pjs_data_plan.md",
    "基础设施/归档/v5草稿/",
    "基础设施/归档/设计材料/",
    "基础设施/归档/项目总结/",
)
EXCLUDED = ("/examples/", "/__pycache__/", "/simply_jp/train_")
SELECTED_JSON = (
    "基础设施/v5/V5P/vocab.json",
    "基础设施/v5/V5P/V5PgO/listening_results_20260924.json",
)
PRIVATE_JSON_KEYS = {
    "resource_root",
    "blind_dir",
    "key_file",
}
HASH = re.compile(r"\b[a-fA-F0-9]{64}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
WIN_PATH = re.compile(r"(?i)(?<![\w])(?:[A-Z]:[\\/]|file:///)[^`\s\"'<>|)\]]+")
UNIX_PATH = re.compile(
    r"(?<![\w])/(?:home|mnt|data|tmp|opt|var|srv|root|Users|media|workspace)/"
    r"[^`\s\"'<>|)\]}，。；、]+"
)
CLOUD_PATH = re.compile(r"(?i)(?<!\w)/(?:TEMP|V5)/[^`\s\"'<>|)\]}，。；、]+")
URL = re.compile(r"https?://[^\s`<>\"')\]]+", re.I)
PLACEHOLDER_LINK = re.compile(
    r"\[([^\]]+)\]\(\s*(?:\$\{(?:LOCAL_PATH|PRIVATE_PATH|PRIVATE_URL)\})\s*\)"
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")
SECRET = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?key|secret[_-]?key|password|bearer)\b"
    r"\s*[:=]\s*([^\s`]+)"
)


def sanitize(text: str) -> str:
    text = HASH.sub("SHA256_REDACTED", text)
    text = EMAIL.sub("CONTACT_REDACTED", text)
    text = IP.sub("IP_REDACTED", text)
    text = WIN_PATH.sub("${LOCAL_PATH}", text)
    text = UNIX_PATH.sub("${PRIVATE_PATH}", text)
    text = CLOUD_PATH.sub("${CLOUD_ARTIFACT}", text)
    text = re.sub(r"(?i)\b(?:jbbj|daodao)\b", "USER", text)
    text = SECRET.sub(lambda m: m.group(0).replace(m.group(1), "REDACTED"), text)

    def replace_url(match: re.Match[str]) -> str:
        value = match.group(0)
        host = re.match(r"https?://([^/:?#]+)", value, re.I).group(1).lower()
        if host in {"localhost", "127.0.0.1"} or "aliyun" in host:
            return "${PRIVATE_URL}"
        if re.search(r"(?i)[?&](?:token|key|signature|auth|x-amz-)", value):
            return "${PRIVATE_URL}"
        return value

    text = URL.sub(replace_url, text)
    return PLACEHOLDER_LINK.sub(lambda match: f"`{match.group(1)}`", text)


def selected(relative: str) -> bool:
    return (
        relative.endswith(".md")
        and any(
            relative == prefix or relative.startswith(prefix)
            for prefix in SELECTED
        )
        and not any(exclusion in relative for exclusion in EXCLUDED)
    )


def repair_missing_links(text: str, document: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        label, link = match.groups()
        if link.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)
        if link.startswith("${") or re.match(r"(?i)^[a-z]:[/\\]", link):
            return f"`{label}`"
        path = link.split("#", 1)[0]
        if not path or path.isdecimal():
            return match.group(0)
        return match.group(0) if (document.parent / path).exists() else f"`{label}`"

    return MARKDOWN_LINK.sub(replace, text)


def sanitize_json(value):
    if isinstance(value, dict):
        return {
            key: sanitize_json(item)
            for key, item in value.items()
            if key not in PRIVATE_JSON_KEYS and not key.endswith("_sha256")
        }
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, str):
        return sanitize(value)
    return value


def writable(target: Path, destination: Path, refresh: bool) -> bool:
    if not target.exists():
        return True
    if not refresh:
        return False
    tracked = subprocess.run(
        ["git", "-C", str(destination.parent), "ls-files", "--error-unmatch",
         "--", str(target)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return tracked.returncode != 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--refresh-untracked",
        action="store_true",
        help="Regenerate selected files only if Git does not track them",
    )
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("source and destination must be separate trees")
    created: list[Path] = []
    for original in source.rglob("*.md"):
        relative = original.relative_to(source).as_posix()
        if not selected(relative):
            continue
        target = destination / relative
        if not writable(target, destination, args.refresh_untracked):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        text = sanitize(original.read_text(encoding="utf-8"))
        notice = (
            "> Public sanitized snapshot. Local paths, artifact locations, and "
            "fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.\n\n"
        )
        target.write_text(notice + text, encoding="utf-8")
        created.append(target)
        print(relative)
    for target in created:
        text = target.read_text(encoding="utf-8")
        text = repair_missing_links(text, target)
        text = "\n".join(line.rstrip() for line in text.splitlines()).rstrip("\n") + "\n"
        target.write_text(text, encoding="utf-8")
    json_count = 0
    for relative in SELECTED_JSON:
        original, target = source / relative, destination / relative
        if not writable(target, destination, args.refresh_untracked):
            continue
        data = json.loads(original.read_text(encoding="utf-8"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(sanitize_json(data), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        json_count += 1
    print(
        f"wrote {len(created)} sanitized Markdown files and {json_count} JSON files;"
        " review before publication"
    )


if __name__ == "__main__":
    main()
