
import os
import re
import argparse
import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm

MAPPING_PATH = "/home/baschmid/sw/ML/ML_FoCal_FinalProject/Barbara/channelmapping.channelmapping"

mapping_df = pd.read_csv(MAPPING_PATH, sep="\t")


def mapped_half_channel_to_raw(asic, half, ch):
    base = 76 * asic + 38 * half

    if ch < 18:
        return base + ch + 1
    else:
        return base + ch + 2


def build_geometry_arrays():
    row_pos = np.full(304, np.nan)
    col_pos = np.full(304, np.nan)

    for _, row in mapping_df.iterrows():
        vldb = int(row["VLDB"])
        asic = int(row["ASIC"])
        half = int(row["HALF"])
        mapped_ch = int(row["CHANNEL"])

        raw_ch = mapped_half_channel_to_raw(asic, half, mapped_ch)

        global_ch = 152 * vldb + raw_ch

        row_pos[global_ch] = int(row["ROW"])
        col_pos[global_ch] = int(row["COL"])

    valid_geom = ~np.isnan(row_pos)

    return row_pos, col_pos, valid_geom


ROW_POS, COL_POS, VALID_GEOM = build_geometry_arrays()



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


    # Physical shower position using channel mapping
    weight = adc_signal_ch.clip(min=0)

    valid = VALID_GEOM & (weight > 0)

    if weight[valid].sum() > 0:
        row_cog = np.average(ROW_POS[valid], weights=weight[valid])
        col_cog = np.average(COL_POS[valid], weights=weight[valid])

        row_width = np.sqrt(
            np.average((ROW_POS[valid] - row_cog) ** 2, weights=weight[valid])
        )
        col_width = np.sqrt(
            np.average((COL_POS[valid] - col_cog) ** 2, weights=weight[valid])
        )

        radial_width = np.sqrt(row_width**2 + col_width**2)

    else:
        row_cog = np.nan
        col_cog = np.nan
        row_width = np.nan
        col_width = np.nan
        radial_width = np.nan

    feats["row_cog_adc"] = float(row_cog)
    feats["col_cog_adc"] = float(col_cog)
    feats["row_width_adc"] = float(row_width)
    feats["col_width_adc"] = float(col_width)
    feats["radial_width_adc"] = float(radial_width)


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
    parser.add_argument("--input-dir", required=True, help="Folder containing ROOT files")
    parser.add_argument("--output", default="energy_training_table.parquet")
    parser.add_argument("--tree", default="data_tree")
    parser.add_argument("--max-events", type=int, default=None)

    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    all_dfs = []

    # Select only the runs we want
    meta_selected = meta[
        (meta["particle"] == "h") &
        (meta["phase_mode"] == "unlocked") &
        (meta["tungsten"] == "no")
    ].copy()

    print("Selected runs from metadata:")
    print(meta_selected[["run", "energy_GeV", "particle", "phase_mode", "tungsten"]])

    for _, metadata_row_series in meta_selected.iterrows():
        run = int(metadata_row_series["run"])

        path = os.path.join(
            args.input_dir,
            f"Run{run}_EventMatch.root"
        )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing ROOT file for run {run}: {path}")

        metadata_row = metadata_row_series.to_dict()

        df = extract_file(
            path,
            metadata_row,
            tree_name=args.tree,
            max_events=args.max_events
        )

        all_dfs.append(df)

    out = pd.concat(all_dfs, ignore_index=True)
    out.to_parquet(args.output, index=False)
    print(f"Saved {len(out)} events to {args.output}")


if __name__ == "__main__":
    main()
