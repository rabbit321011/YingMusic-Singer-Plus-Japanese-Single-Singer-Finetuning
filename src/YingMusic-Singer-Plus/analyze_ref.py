import json, numpy as np

with open("${SERVER_ROOT}/final_sum_large/train_singnet.json") as f:
    train = json.load(f)
with open("${SERVER_ROOT}/final_sum_large/test_singnet.json") as f:
    test = json.load(f)

all_data = train + test
durations = np.array([d["Duration"] for d in all_data])
T_frames = durations * 44100 / 2048  # approx latent frames

print(f"Total: {len(all_data)} samples")
print(f"Duration: min={durations.min():.1f}s  max={durations.max():.1f}s  mean={durations.mean():.1f}s  median={np.median(durations):.1f}s")
print()

# ref_len mechanism
for label in ["Train", "Test", "All"]:
    d = train if label == "Train" else (test if label == "Test" else all_data)
    dur = np.array([x["Duration"] for x in d])
    T = dur * 44100 / 2048
    
    # Mechanism: random 12.5%-33%, but if 5s > 65% of total, fallback to 40-60%
    ref_len_frames = []
    ref_len_sec = []
    fails = 0
    for i, t_frames in enumerate(T):
        low_pct = 0.125; high_pct = 0.33
        if 5.0 > dur[i] * 0.65:
            # total too short for 5s lower bound, use percent only
            ref_frames = t_frames * np.random.uniform(low_pct, high_pct)
        else:
            # try 5s lower bound
            ref_frames_min = min(5.0 * 44100 / 2048, t_frames * high_pct)
            ref_frames_max = t_frames * high_pct
            # Actually reread: "加个下限时长5s，然后如果5s占比大于65%，那么则固定为40%~60%随机而放弃5s下限"
            five_sec_ratio = (5.0 * 44100 / 2048) / t_frames
            if five_sec_ratio > 0.65:
                ref_frames = t_frames * np.random.uniform(0.40, 0.60)
            else:
                ref_frames_min = 5.0 * 44100 / 2048
                ref_frames_max = t_frames * 0.33
                if ref_frames_min > ref_frames_max:
                    ref_frames = t_frames * np.random.uniform(0.40, 0.60)
                else:
                    ref_frames = np.random.uniform(ref_frames_min, ref_frames_max)
        
        ref_len_frames.append(ref_frames)
        ref_len_sec.append(ref_frames * 2048 / 44100)
    
    ref_len_frames = np.array(ref_len_frames)
    ref_len_sec = np.array(ref_len_sec)
    
    print(f"[{label}] {len(d)} samples:")
    print(f"  Duration: {np.percentile(dur, [10,25,50,75,90])}")
    print(f"  ref_len sec: mean={ref_len_sec.mean():.1f}s median={np.median(ref_len_sec):.1f}s")
    print(f"  ref_len%: mean={100*np.mean(ref_len_frames/T):.1f}% median={100*np.median(ref_len_frames/T):.1f}%")
    
    # how many triggered the fallback
    five_sec_frames = 5.0 * 44100 / 2048
    short = T * 0.65 < five_sec_frames
    print(f"  short audio (5s > 65%): {short.sum()}/{len(T)} ({100*short.sum()/len(T):.1f}%)")
    print()

# Check: what % of samples have duration < 5s? < 7.7s?
print("=== Duration buckets ===")
for bound in [3, 5, 7.7, 10, 15, 20, 30]:
    count = (durations < bound).sum()
    print(f"  <{bound:4.0f}s: {count:6d} ({100*count/len(durations):5.1f}%)")

















