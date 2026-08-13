import json
data = json.load(open("${SERVER_ROOT}${CLOUD_ARTIFACT}"))
langs = set(d["language"] for d in data)
print("Languages:", langs)
for lang in ["zh", "en", "ja"]:
    items = [d for d in data if d["language"] == lang]
    print(f"\n{lang}: {len(items)} samples")
    if items:
        print(items[0])

















