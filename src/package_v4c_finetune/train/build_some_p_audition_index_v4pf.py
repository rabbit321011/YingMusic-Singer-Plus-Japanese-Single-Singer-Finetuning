import argparse
import html
import json
from pathlib import Path


def audio_cell(sample_id, name, label):
    source = f"{sample_id}/{name}"
    return (
        f'<div class="audio-label">{html.escape(label)}</div>'
        f'<audio controls preload="none" src="{html.escape(source)}"></audio>'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack_dir", required=True)
    parser.add_argument("--review_version", default="v1")
    parser.add_argument("--timbre", choices=("sine", "piano"), default="sine")
    args = parser.parse_args()

    pack_dir = Path(args.pack_dir)
    report = json.loads((pack_dir / "report_selected.json").read_text(encoding="utf-8"))
    audio_extension = "sine.wav" if args.timbre == "sine" else "piano.ogg"
    timbre_label = "sine" if args.timbre == "sine" else "piano"
    rows = []
    for entry in report["entries"]:
        sample_id = entry["id"]
        model = entry["model"]
        source = html.escape(entry["source_name"])
        stats = (
            f'{entry["decoded_duration"]:.2f}s | MIDI {model["min_midi"]}'
            f'-{model["max_midi"]} | REST {model["rest_ratio"]:.1%} | '
            f'{model["decoded_note_count"]} notes'
        )
        rows.append(
            f'''<tr data-group="{html.escape(entry["group"])}" data-bucket="{html.escape(entry["bucket"])}">
  <td class="identity"><strong>{sample_id}</strong><span>{html.escape(entry["bucket"])}</span></td>
  <td class="source"><div>{source}</div><small>{html.escape(stats)}</small></td>
  <td>{audio_cell(sample_id, "original_mono.wav", "Original")}</td>
  <td>{audio_cell(sample_id, f"p_native_{audio_extension}", f"P native ({timbre_label})")}</td>
  <td>{audio_cell(sample_id, f"p_model_{audio_extension}", f"P model ({timbre_label})")}</td>
  <td class="review">
    <select data-field="verdict" data-id="{sample_id}" aria-label="{sample_id} verdict">
      <option value="">Unreviewed</option>
      <option value="pass">Pass</option>
      <option value="native_bad">Native bad</option>
      <option value="model_bad">Model bad</option>
      <option value="both_bad">Both bad</option>
      <option value="boundary_bad">Boundary bad</option>
    </select>
    <input data-field="notes" data-id="{sample_id}" aria-label="{sample_id} notes" placeholder="Notes">
  </td>
</tr>'''
        )

    document = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V4Pf SOME/P Audition</title>
<style>
:root {{ color-scheme: light; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color: #202124; background: #f5f6f7; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; }}
header {{ position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #cfd3d7; padding: 12px 18px; }}
h1 {{ font-size: 20px; margin: 0 0 10px; font-weight: 650; letter-spacing: 0; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
select, input, button {{ min-height: 34px; border: 1px solid #aeb4ba; border-radius: 4px; background: #fff; color: #202124; font: inherit; padding: 5px 8px; }}
button {{ cursor: pointer; background: #1769aa; color: #fff; border-color: #1769aa; }}
#count {{ font-size: 13px; color: #59636d; margin-left: auto; }}
main {{ padding: 12px; overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; min-width: 1280px; }}
th {{ position: sticky; top: 89px; z-index: 1; background: #e9edf0; text-align: left; font-size: 12px; color: #46515b; }}
th, td {{ padding: 8px; border: 1px solid #d8dde1; vertical-align: middle; }}
tr:nth-child(even) {{ background: #fafbfc; }}
.identity {{ width: 82px; }}
.identity strong, .identity span {{ display: block; }}
.identity span, small {{ color: #65717c; font-size: 11px; }}
.source {{ min-width: 300px; max-width: 420px; overflow-wrap: anywhere; }}
.source small {{ display: block; margin-top: 5px; }}
.audio-label {{ font-size: 11px; color: #59636d; margin-bottom: 3px; }}
audio {{ display: block; width: 220px; height: 32px; }}
.review {{ min-width: 180px; }}
.review select, .review input {{ display: block; width: 100%; margin: 3px 0; }}
.hidden {{ display: none; }}
</style>
</head>
<body>
<header>
  <h1>V4Pf SOME/P Audition</h1>
  <div class="toolbar">
    <select id="group" aria-label="Group filter">
      <option value="">All groups</option>
      <option value="non_hanamaru">Non-Hanamaru</option>
      <option value="hanamaru_candidate">Hanamaru</option>
    </select>
    <select id="bucket" aria-label="Duration bucket filter">
      <option value="">All durations</option>
      <option value="0-10">0-10s</option>
      <option value="10-20">10-20s</option>
      <option value="20-30">20-30s</option>
      <option value="external">External</option>
    </select>
    <input id="search" type="search" placeholder="Search ID or source" aria-label="Search">
    <button id="export" type="button">Export JSON</button>
    <span id="count"></span>
  </div>
</header>
<main>
<table>
  <thead><tr><th>ID</th><th>Source / stats</th><th>Original</th><th>P native (~86Hz, {timbre_label})</th><th>P model (21.53Hz, {timbre_label})</th><th>Review</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
</main>
<script>
const storageKey = "v4pf_some_p_audition_{html.escape(args.review_version)}";
const state = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
const controls = [...document.querySelectorAll("[data-field]")];
for (const control of controls) {{
  const id = control.dataset.id;
  const field = control.dataset.field;
  control.value = state[id]?.[field] || "";
  control.addEventListener("change", () => {{
    state[id] ||= {{}};
    state[id][field] = control.value;
    localStorage.setItem(storageKey, JSON.stringify(state));
  }});
}}
function applyFilters() {{
  const group = document.querySelector("#group").value;
  const bucket = document.querySelector("#bucket").value;
  const query = document.querySelector("#search").value.trim().toLowerCase();
  let visible = 0;
  for (const row of document.querySelectorAll("tbody tr")) {{
    const matches = (!group || row.dataset.group === group) && (!bucket || row.dataset.bucket === bucket) && (!query || row.textContent.toLowerCase().includes(query));
    row.classList.toggle("hidden", !matches);
    if (matches) visible++;
  }}
  document.querySelector("#count").textContent = `${{visible}} / 62`;
}}
for (const id of ["group", "bucket", "search"]) document.querySelector(`#${{id}}`).addEventListener("input", applyFilters);
document.querySelector("#export").addEventListener("click", () => {{
  const blob = new Blob([JSON.stringify({{schema:"v4pf_some_p_audition_review_{html.escape(args.review_version)}", reviews:state}}, null, 2)], {{type:"application/json"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "v4pf_some_p_audition_review.json";
  link.click();
  URL.revokeObjectURL(link.href);
}});
applyFilters();
</script>
</body>
</html>
'''
    (pack_dir / "index.html").write_text(document, encoding="utf-8")
    print(pack_dir / "index.html")


if __name__ == "__main__":
    main()
