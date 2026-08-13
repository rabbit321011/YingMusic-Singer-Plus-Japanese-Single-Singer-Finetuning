import json
with open("${SERVER_ROOT}/final_sum_large/train_singnet.json") as f:
    data = json.load(f)

for i in range(5):
    print(f"[{i}] {data[i]['Text'][:80]}")
print()

pipe_count = sum(1 for d in data if "|" in d.get("Text",""))
nl_count = sum(1 for d in data if "\n" in d.get("Text",""))
lens = [len(d["Text"]) for d in data]
print(f"With '|': {pipe_count}/{len(data)}")
print(f"With newline: {nl_count}/{len(data)}")
print(f"Text len: min={min(lens)} max={max(lens)} avg={sum(lens)//len(lens)}")

# check: is | ever used?
if pipe_count:
    sample = [d for d in data if "|" in d["Text"]][0]
    print(f"Sample with |: {sample['Text'][:100]}")

















