import json

d = json.load(open("${REMOTE_ROOT}/final_sum_large/train_singnet.json"))
print("Total records:", len(d))
print("First 5 samples:")
for r in d[:5]:
    print(f"  [{r['Language']}] {r['Text'][:80]}")

langs = {}
for r in d:
    l = r["Language"]
    langs[l] = langs.get(l, 0) + 1
print("\nLanguage distribution:")
for k, v in sorted(langs.items()):
    print(f"  {k}: {v} ({100*v/len(d):.1f}%)")

