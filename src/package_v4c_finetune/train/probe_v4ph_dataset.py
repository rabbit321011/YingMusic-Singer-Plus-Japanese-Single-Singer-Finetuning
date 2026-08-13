import argparse

from train_v4ph import HDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest_sha256", required=True)
    parser.add_argument("--h_config_fingerprint", required=True)
    parser.add_argument("--game_cache_manifest", required=True)
    parser.add_argument("--audio_root_override", default=None)
    args = parser.parse_args()

    dataset = HDataset(
        args.manifest,
        args.manifest_sha256,
        args.h_config_fingerprint,
        game_cache_manifest=args.game_cache_manifest,
        audio_root_override=args.audio_root_override,
    )
    probes = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for index in probes:
        sample = dataset[index]
        print(
            f"index={index} sample={sample['sample_id']} "
            f"wav={tuple(sample['wav'].shape)} sr={sample['sr']} "
            f"cache_notes={sample['game_cache']['durations'].shape[0]}"
        )
    print(f"V4PH dataset probe passed: entries={len(dataset)}")


if __name__ == "__main__":
    main()
