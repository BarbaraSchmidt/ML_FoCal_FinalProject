
import os
import re
import argparse
import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm


def run_from_filename(path):
    m = re.search(r"Run(\d+)", os.path.basename(path))
    if not m:
        raise ValueError(f"Could not extract run number from {path}")
    return int(m.group(1))


def decode_tot(arr):
    """
    H2GCROC ToT convention:
    if value > 511, mask to 10 bits and shift left by 3.
    """
    arr = arr.astype(np.int64, copy=True)
    mask = arr > 511
    arr[mask] = (arr[mask] & 0b0111111111) << 3
    return arr


def event_features(val0_fpga0, val0_fpga1, val1_fpga0, val1_fpga1, mg):
    """
    Returns simple event-level features.
    Assumes flattened arrays with shape mg * 152 per FPGA.
    """
    adc0 = np.asarray(val0_fpga0).reshape(mg, 152)
    adc1 = np.asarray(val0_fpga1).reshape(mg, 152)

    tot0 = decode_tot(np.asarray(val1_fpga0)).reshape(mg, 152)
    tot1 = decode_tot(np.asarray(val1_fpga1)).reshape(mg, 152)

    adc = np.concatenate([adc0, adc1], axis=1)  # shape: mg x 304
    tot = np.concatenate([tot0, tot1], axis=1)

    pedestal_adc = adc[:3].min(axis=0)

    adc_max_ch_raw = adc.max(axis=0)
    adc_signal_ch = adc_max_ch_raw - pedestal_adc
    adc_signal_ch = np.clip(adc_signal_ch, 0, None)

    tot_max_ch = tot.max(axis=0)

    # Basic robust first version
    feats = {
        "adc_sum": float(adc_signal_ch.sum()),
        "tot_sum": float(tot_max_ch.sum()),
        "adc_max": float(adc_signal_ch.max()),
        "tot_max": float(tot_max_ch.max()),
        "adc_mean": float(adc_signal_ch.mean()),
        "tot_mean": float(tot_max_ch.mean()),
        "n_adc_hit_50": int((adc_signal_ch > 50).sum()),
        "n_adc_hit_100": int((adc_signal_ch > 100).sum()),
        "n_tot_hit": int((tot_max_ch > 0).sum()),
        "n_adc_saturated": int((adc_max_ch_raw >= 1022).sum()),
        "n_tot_saturated": int((tot_max_ch >= 4088).sum()),
        "adc_raw_sum": float(adc_max_ch_raw.sum()),
        "adc_pedestal_mean": float(pedestal_adc.mean()),
    }

    # Per-FPGA sums
    feats["adc_sum_fpga0"] = float(adc_signal_ch[:152].sum())
    feats["adc_sum_fpga1"] = float(adc_signal_ch[152:].sum())
    feats["tot_sum_fpga0"] = float(tot_max_ch[:152].sum())
    feats["tot_sum_fpga1"] = float(tot_max_ch[152:].sum())

    # Crude position from hit pattern, not external metadata
    channels = np.arange(304)
    weight = adc_signal_ch.clip(min=0)
    if weight.sum() > 0:
        feats["channel_cog_adc"] = float(np.average(channels, weights=weight))
        feats["channel_width_adc"] = float(np.sqrt(np.average((channels - feats["channel_cog_adc"])**2, weights=weight)))
    else:
        feats["channel_cog_adc"] = np.nan
        feats["channel_width_adc"] = np.nan

    return feats


def get_machinegun_count(root_file, default=None):
    """
    Try common metadata keys. Falls back to default or inferred size later.
    """
    for key in ["CorrectMachinegunCount", "EventRecon_machine_gun_samples"]:
        if key in root_file:
            obj = root_file[key]
            try:
                return int(obj.member("fTitle"))
            except Exception:
                try:
                    return int(str(obj))
                except Exception:
                    pass
    return default


def extract_file(path, metadata_row, tree_name="data_tree", max_events=None):
    run = run_from_filename(path)

    with uproot.open(path) as f:
        tree = f[tree_name]
        mg = get_machinegun_count(f)

        needed = ["val0_list_0", "val0_list_1", "val1_list_0", "val1_list_1"]
        missing = [b for b in needed if b not in tree.keys()]
        if missing:
            raise KeyError(f"Missing branches in {path}: {missing}. Available branches: {tree.keys()}")

        rows = []
        branches = tree.iterate(needed, step_size=1000, library="np")

        n_done = 0
        for batch in tqdm(branches, desc=f"Run {run}"):
            n_batch = len(batch["val0_list_0"])

            for i in range(n_batch):
                if max_events is not None and n_done >= max_events:
                    break

                # Infer MG count if metadata was unavailable
                mg_i = mg
                if mg_i is None:
                    mg_i = len(batch["val0_list_0"][i]) // 152

                feats = event_features(
                    batch["val0_list_0"][i],
                    batch["val0_list_1"][i],
                    batch["val1_list_0"][i],
                    batch["val1_list_1"][i],
                    mg_i,
                )

                row = {"run": run, "event_index": n_done}
                row.update(metadata_row)
                row.update(feats)
                rows.append(row)
                n_done += 1

            if max_events is not None and n_done >= max_events:
                break

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="CSV with run-level metadata")
    parser.add_argument("--input", nargs="+", required=True, help="ROOT files")
    parser.add_argument("--output", default="energy_training_table.parquet")
    parser.add_argument("--tree", default="data_tree")
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    all_dfs = []

    for path in args.input:
        run = run_from_filename(path)
        matches = meta[meta["run"] == run]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one metadata row for run {run}, got {len(matches)}")

        metadata_row = matches.iloc[0].to_dict()
        df = extract_file(path, metadata_row, tree_name=args.tree, max_events=args.max_events)
        all_dfs.append(df)

    out = pd.concat(all_dfs, ignore_index=True)
    out.to_parquet(args.output, index=False)
    print(f"Saved {len(out)} events to {args.output}")


if __name__ == "__main__":
    main()
