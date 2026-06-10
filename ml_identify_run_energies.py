import csv
import os
import re
import warnings
from pathlib import Path

import ROOT
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT.gROOT.SetBatch(True)

# =========================================================
# EASY SETTINGS
# =========================================================


DATA_DIR = Path(".")
OUT_DIR = Path("2025_ml_energy_identification_all_runs")
CACHE_DIR = OUT_DIR / "feature_cache"

OUT_FIG = OUT_DIR / "ml_energy_identification_two_figures_no_h47_eventmatch.png"
OUT_RESULTS_CSV = OUT_DIR / "ml_energy_identification_results_no_h47_eventmatch.csv"
OUT_LOO_CSV = OUT_DIR / "ml_energy_identification_leave_one_run_out_no_h47_eventmatch.csv"
OUT_GROUPS_CSV = OUT_DIR / "ml_energy_identification_group_status_no_h47_eventmatch.csv"

# If True, missing ROOT files are skipped.
# If False, the script stops when one file is missing.
SKIP_MISSING_FILES = True

# Feature cache makes re-running much faster after the first full pass.
USE_FEATURE_CACHE = True
FORCE_REBUILD_CACHE = False

EVENT_MAX = 32900
SHOW_PROGRESS = True
RANDOM_STATE = 42

N_SAMPLES = 16
HALVES_PER_VLDB = 4
SLOTS_PER_HALF = 38
VALUES_PER_SAMPLE = HALVES_PER_VLDB * SLOTS_PER_HALF

USE_PEDESTAL_SUBTRACTION = True
CLIP_NEGATIVE_AFTER_PEDESTAL = True

PEAK_SAMPLES = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32)
PEDESTAL_SAMPLES = np.array([0, 1, 2], dtype=np.int32)


RUN_TABLE = {
    # Hadron, bias 54.5 V, standard reconstruction
    445: {"energy": 150, "particle": "h", "bias_v": 54.5, "optical": False},
    446: {"energy": 200, "particle": "h", "bias_v": 54.5, "optical": False},
    448: {"energy": 300, "particle": "h", "bias_v": 54.5, "optical": False},
    450: {"energy": 100, "particle": "h", "bias_v": 54.5, "optical": False},
    451: {"energy": 80,  "particle": "h", "bias_v": 54.5, "optical": False},
    452: {"energy": 60,  "particle": "h", "bias_v": 54.5, "optical": False},
    458: {"energy": 250, "particle": "h", "bias_v": 54.5, "optical": False},
    459: {"energy": 350, "particle": "h", "bias_v": 54.5, "optical": False},

    # Hadron, bias 43 V, EventMatch
    577: {"energy": 60,  "particle": "h", "bias_v": 43, "optical": False},
    578: {"energy": 60,  "particle": "h", "bias_v": 43, "optical": False},
    583: {"energy": 200, "particle": "h", "bias_v": 43, "optical": False},
    584: {"energy": 200, "particle": "h", "bias_v": 43, "optical": False},
    585: {"energy": 100, "particle": "h", "bias_v": 43, "optical": False},
    586: {"energy": 100, "particle": "h", "bias_v": 43, "optical": False},
    591: {"energy": 150, "particle": "h", "bias_v": 43, "optical": False},
    592: {"energy": 150, "particle": "h", "bias_v": 43, "optical": False},
    593: {"energy": 300, "particle": "h", "bias_v": 43, "optical": False},
    594: {"energy": 300, "particle": "h", "bias_v": 43, "optical": False},
    687: {"energy": 325, "particle": "h", "bias_v": 43, "optical": False},

    # Hadron, bias 45 V, EventMatch
    579: {"energy": 60,  "particle": "h", "bias_v": 45, "optical": False},
    580: {"energy": 60,  "particle": "h", "bias_v": 45, "optical": False},
    581: {"energy": 200, "particle": "h", "bias_v": 45, "optical": False},
    582: {"energy": 200, "particle": "h", "bias_v": 45, "optical": False},
    587: {"energy": 100, "particle": "h", "bias_v": 45, "optical": False},
    588: {"energy": 100, "particle": "h", "bias_v": 45, "optical": False},
    589: {"energy": 150, "particle": "h", "bias_v": 45, "optical": False},
    590: {"energy": 150, "particle": "h", "bias_v": 45, "optical": False},
    595: {"energy": 300, "particle": "h", "bias_v": 45, "optical": False},
    596: {"energy": 300, "particle": "h", "bias_v": 45, "optical": False},
    685: {"energy": 325, "particle": "h", "bias_v": 45, "optical": False},
    686: {"energy": 325, "particle": "h", "bias_v": 45, "optical": False},

    # Electron, bias 43 V, EventMatch
    605: {"energy": 60,  "particle": "e", "bias_v": 43, "optical": False},
    606: {"energy": 60,  "particle": "e", "bias_v": 43, "optical": False},
    607: {"energy": 60,  "particle": "e", "bias_v": 43, "optical": False},
    610: {"energy": 80,  "particle": "e", "bias_v": 43, "optical": False},
    611: {"energy": 80,  "particle": "e", "bias_v": 43, "optical": False},
    614: {"energy": 100, "particle": "e", "bias_v": 43, "optical": False},
    615: {"energy": 100, "particle": "e", "bias_v": 43, "optical": False},

    # Electron, bias 45 V, EventMatch
    603: {"energy": 60,  "particle": "e", "bias_v": 45, "optical": False},
    604: {"energy": 60,  "particle": "e", "bias_v": 45, "optical": False},
    608: {"energy": 80,  "particle": "e", "bias_v": 45, "optical": False},
    609: {"energy": 80,  "particle": "e", "bias_v": 45, "optical": False},
    612: {"energy": 100, "particle": "e", "bias_v": 45, "optical": False},
    613: {"energy": 100, "particle": "e", "bias_v": 45, "optical": False},

    # Hadron, bias 47 V, EventMatch, no optical
    650: {"energy": 350, "particle": "h", "bias_v": 47, "optical": False},
    651: {"energy": 350, "particle": "h", "bias_v": 47, "optical": False},
    652: {"energy": 60,  "particle": "h", "bias_v": 47, "optical": False},
    660: {"energy": 150, "particle": "h", "bias_v": 47, "optical": False},
    661: {"energy": 150, "particle": "h", "bias_v": 47, "optical": False},

    # Hadron, bias 47 V, EventMatch, with optical
    653: {"energy": 60,  "particle": "h", "bias_v": 47, "optical": True},
    655: {"energy": 200, "particle": "h", "bias_v": 47, "optical": True},
    656: {"energy": 200, "particle": "h", "bias_v": 47, "optical": True},
    657: {"energy": 100, "particle": "h", "bias_v": 47, "optical": True},
    658: {"energy": 100, "particle": "h", "bias_v": 47, "optical": True},
    659: {"energy": 150, "particle": "h", "bias_v": 47, "optical": True},
    662: {"energy": 150, "particle": "h", "bias_v": 47, "optical": True},
}

# Runs removed from this version:
#   h, bias 47 V, EventMatch, no optical
# This is the group previously shown as h/47V/EM.
EXCLUDED_RUNS = {650, 651, 652, 660, 661}

# =========================================================
# CHANNEL SELECTION
# =========================================================

CONNECTED_CHANNELS = set(
    list(range(0, 8)) +
    list(range(9, 17)) +
    list(range(18, 26)) +
    list(range(27, 35))
)

USED_HALVES_BY_VLDB = {
    0: {0, 1},
    1: {0, 1, 2, 3},
}

BAD_CHANNELS_BY_VLDB = {
    1: {(0, 30)},
}

# =========================================================
# STYLE
# =========================================================

plt.rcParams.update({
    "figure.facecolor": "0.94",
    "axes.facecolor": "0.94",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.titlesize": 16,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =========================================================
# PATHS AND RUN HELPERS
# =========================================================

def candidate_data_dirs():
    dirs = []
    for d in [DATA_DIR, DATA_DIR / "2025_data", Path("2025_data")]:
        d = Path(d)
        if d not in dirs:
            dirs.append(d)
    return dirs


def run_to_path(run_number: int) -> Path:
    """
    Accept both filename types:
      Run0445.root
      Run577_EventMatch.root
    The script checks DATA_DIR, DATA_DIR/2025_data, and ./2025_data.
    """
    names = [
        f"Run{run_number:04d}.root",
        f"Run{run_number}.root",
        f"Run{run_number}_EventMatch.root",
        f"Run{run_number:04d}_EventMatch.root",
    ]

    candidates = []
    for d in candidate_data_dirs():
        for name in names:
            candidates.append(d / name)

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def get_run_number(path):
    match = re.search(r"Run0*(\d+)(?:_EventMatch)?\.root$", str(path))
    return int(match.group(1)) if match else -1


def reconstruction_chain(path):
    name = Path(path).name
    if "EventMatch" in name:
        return "EventMatch"
    return "standard"


def bias_to_text(value):
    value = float(value)
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.1f}"


def group_key(run):
    return (run["particle"], float(run["bias_v"]), run["chain"], bool(run.get("optical", False)))


def group_text(key):
    particle, bias_v, chain, optical = key
    optical_text = "optical" if optical else "no optical"
    return f"{particle}, bias {bias_to_text(bias_v)} V, {chain}, {optical_text}"


def group_short_text(key):
    particle, bias_v, chain, optical = key
    chain_short = "EM" if chain == "EventMatch" else "std"
    optical_short = "/opt" if optical else ""
    return f"{particle}/{bias_to_text(bias_v)}V/{chain_short}{optical_short}"


def group_sort_key(key):
    particle, bias_v, chain, optical = key
    return (particle, float(bias_v), chain, optical)


def open_tree(path):
    root_file = ROOT.TFile(str(path), "READ")
    tree = root_file.Get("data_tree;2") or root_file.Get("data_tree")

    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find data_tree in {path}")

    return root_file, tree


def channel_to_slot_in_half(channel):
    if not (0 <= channel <= 35):
        raise ValueError(f"Invalid channel: {channel}")
    return channel + 1 if channel <= 17 else channel + 2


def half_and_channel_to_raw_index(half_global, channel):
    return half_global * SLOTS_PER_HALF + channel_to_slot_in_half(channel)


def build_selected_indices(vldb):
    indices = []
    used_halves = USED_HALVES_BY_VLDB.get(vldb, {0, 1, 2, 3})
    bad_channels = BAD_CHANNELS_BY_VLDB.get(vldb, set())

    for half in range(HALVES_PER_VLDB):
        if half not in used_halves:
            continue

        for channel in sorted(CONNECTED_CHANNELS):
            if (half, channel) in bad_channels:
                continue

            indices.append(half_and_channel_to_raw_index(half, channel))

    return np.array(indices, dtype=np.int32)


SELECTED_IDX_VLDB0 = build_selected_indices(0)
SELECTED_IDX_VLDB1 = build_selected_indices(1)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def get_event_arrays(val0_list_0, val0_list_1):
    event0 = np.asarray(val0_list_0, dtype=np.float64).reshape(
        N_SAMPLES,
        VALUES_PER_SAMPLE,
    )

    event1 = np.asarray(val0_list_1, dtype=np.float64).reshape(
        N_SAMPLES,
        VALUES_PER_SAMPLE,
    )

    return event0, event1


def robust_spread(values):
    """Robust event-to-event spread using MAD converted to sigma-like units."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(1.4826 * mad)


def nearest_allowed_energy(pred_energy, allowed_energies):
    allowed_energies = np.asarray(sorted(set(allowed_energies)), dtype=float)
    idx = np.argmin(np.abs(allowed_energies - pred_energy))
    return float(allowed_energies[idx])


def event_ml_features(list0, list1):
    """
    Convert one ROOT event into ML features.

    Same core observable as before:
      pedestal subtract samples 0-2,
      use signal samples 3-11,
      take channel-wise max,
      sum over selected channels.

    Extra time-shape features are added, but the model is now trained separately
    per particle, bias voltage, and reconstruction chain.
    """
    event0, event1 = get_event_arrays(list0, list1)

    sel0 = event0[:, SELECTED_IDX_VLDB0]
    sel1 = event1[:, SELECTED_IDX_VLDB1]

    signal0 = sel0.copy()
    signal1 = sel1.copy()

    if USE_PEDESTAL_SUBTRACTION:
        pedestal0 = np.mean(sel0[PEDESTAL_SAMPLES, :], axis=0)
        pedestal1 = np.mean(sel1[PEDESTAL_SAMPLES, :], axis=0)
        signal0 = signal0 - pedestal0[None, :]
        signal1 = signal1 - pedestal1[None, :]

    if CLIP_NEGATIVE_AFTER_PEDESTAL:
        signal0 = np.maximum(signal0, 0.0)
        signal1 = np.maximum(signal1, 0.0)

    peak0 = signal0[PEAK_SAMPLES, :]
    peak1 = signal1[PEAK_SAMPLES, :]

    adc_peakmax = float(
        np.sum(np.max(peak0, axis=0)) +
        np.sum(np.max(peak1, axis=0))
    )

    sample_sums = np.sum(peak0, axis=1) + np.sum(peak1, axis=1)
    total_integral = float(np.sum(sample_sums))
    max_sample_sum = float(np.max(sample_sums))
    argmax_sample = float(PEAK_SAMPLES[np.argmax(sample_sums)])

    safe_total = max(total_integral, 1e-12)
    weighted_sample = float(np.sum(PEAK_SAMPLES * sample_sums) / safe_total)

    early_frac = float(np.sum(sample_sums[0:3]) / safe_total)    # samples 3,4,5
    middle_frac = float(np.sum(sample_sums[3:6]) / safe_total)   # samples 6,7,8
    late_frac = float(np.sum(sample_sums[6:9]) / safe_total)     # samples 9,10,11

    features = [
        adc_peakmax,
        np.log1p(adc_peakmax),
        total_integral,
        np.log1p(total_integral),
        max_sample_sum,
        np.log1p(max_sample_sum),
        argmax_sample,
        weighted_sample,
        early_frac,
        middle_frac,
        late_frac,
    ]

    features.extend(sample_sums.tolist())
    features.extend(np.log1p(sample_sums).tolist())

    return np.asarray(features, dtype=np.float64)


FEATURE_NAMES = (
    [
        "adc_peakmax",
        "log_adc_peakmax",
        "total_integral",
        "log_total_integral",
        "max_sample_sum",
        "log_max_sample_sum",
        "argmax_sample",
        "weighted_sample",
        "early_frac",
        "middle_frac",
        "late_frac",
    ]
    + [f"sample_sum_{s}" for s in PEAK_SAMPLES]
    + [f"log_sample_sum_{s}" for s in PEAK_SAMPLES]
)

# =========================================================
# LOAD RUNS WITH OPTIONAL CACHE
# =========================================================

def cache_path_for_run(run_number, chain, event_max):
    safe_chain = chain.replace("/", "_")
    return CACHE_DIR / f"Run{run_number:04d}_{safe_chain}_eventmax{event_max}.npz"


def try_load_cache(path, run_number, chain, event_max):
    if not USE_FEATURE_CACHE or FORCE_REBUILD_CACHE:
        return None

    cache_path = cache_path_for_run(run_number, chain, event_max)
    if not cache_path.exists():
        return None

    try:
        stat = path.stat()
        data = np.load(cache_path, allow_pickle=False)
        if int(data["file_size"]) != int(stat.st_size):
            return None
        if abs(float(data["file_mtime"]) - float(stat.st_mtime)) > 1e-6:
            return None
        X = data["X"]
        print(f"Loaded cached features: {cache_path}")
        return X
    except Exception as exc:
        warnings.warn(f"Could not load cache {cache_path}: {exc}")
        return None


def save_cache(path, run_number, chain, event_max, X):
    if not USE_FEATURE_CACHE:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for_run(run_number, chain, event_max)
    stat = path.stat()
    np.savez_compressed(
        cache_path,
        X=X,
        run_number=int(run_number),
        chain=chain,
        event_max=int(event_max),
        file_size=int(stat.st_size),
        file_mtime=float(stat.st_mtime),
    )
    print("Saved feature cache:", cache_path)


def load_single_run(path, event_max):
    run_number = get_run_number(path)
    info = RUN_TABLE[run_number]
    chain = reconstruction_chain(path)

    print(
        f"Loading {path} -> Run {run_number}, "
        f"{info['energy']} GeV, {info['particle']}, bias {bias_to_text(info['bias_v'])} V, {chain}, optical={info.get('optical', False)}"
    )

    X_cached = try_load_cache(path, run_number, chain, event_max)
    if X_cached is not None:
        return {
            "run": int(run_number),
            "energy": float(info["energy"]),
            "particle": info["particle"],
            "bias_v": float(info["bias_v"]),
            "optical": bool(info.get("optical", False)),
            "chain": chain,
            "path": str(path),
            "X": X_cached,
        }

    root_file, tree = open_tree(path)
    n_events = min(event_max, int(tree.GetEntries()))

    if n_events == 0:
        root_file.Close()
        raise RuntimeError(f"No events found in {path}")

    first_features = None
    X = None

    iterator = range(n_events)
    if SHOW_PROGRESS:
        iterator = tqdm(iterator, desc=f"Run {run_number}")

    for i in iterator:
        tree.GetEntry(i)
        features = event_ml_features(tree.val0_list_0, tree.val0_list_1)

        if first_features is None:
            first_features = features
            X = np.zeros((n_events, len(first_features)), dtype=np.float64)

        X[i, :] = features

    root_file.Close()
    save_cache(path, run_number, chain, event_max, X)

    return {
        "run": int(run_number),
        "energy": float(info["energy"]),
        "particle": info["particle"],
        "bias_v": float(info["bias_v"]),
            "optical": bool(info.get("optical", False)),
        "chain": chain,
        "path": str(path),
        "X": X,
    }


def load_all_runs():
    runs = []

    print(f"VLDB 0: using {len(SELECTED_IDX_VLDB0)} connected channels per sample")
    print(f"VLDB 1: using {len(SELECTED_IDX_VLDB1)} connected channels per sample")
    print(
        "Total : using "
        f"{len(SELECTED_IDX_VLDB0) + len(SELECTED_IDX_VLDB1)} "
        "connected channels per sample"
    )
    print("Data directories checked:", ", ".join(str(d) for d in candidate_data_dirs()))

    for run_number in sorted(RUN_TABLE):
        if run_number in EXCLUDED_RUNS:
            print(f"Skipping Run {run_number}: excluded h, 47 V, EventMatch, no optical group")
            continue

        path = run_to_path(run_number)

        if not path.exists():
            msg = f"Missing file for Run {run_number}: checked names like {path}"
            if SKIP_MISSING_FILES:
                warnings.warn(msg + " -> skipped")
                continue
            raise FileNotFoundError(msg)

        runs.append(load_single_run(path, EVENT_MAX))

    if len(runs) == 0:
        raise RuntimeError("No runs were loaded. Check DATA_DIR and file names.")

    return runs

# =========================================================
# GROUPED ML MODEL
# =========================================================

def make_model():
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=450,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=0.02,
        random_state=RANDOM_STATE,
    )


def stack_events(runs):
    X_list = []
    y_list = []
    group_list = []

    for run in runs:
        X = run["X"]
        y = np.full(X.shape[0], run["energy"], dtype=np.float64)
        groups = np.full(X.shape[0], run["run"], dtype=np.int32)

        X_list.append(X)
        y_list.append(y)
        group_list.append(groups)

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    groups_all = np.concatenate(group_list)

    return X_all, y_all, groups_all


def split_runs_by_group(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault(group_key(run), []).append(run)
    return {k: sorted(v, key=lambda r: r["run"]) for k, v in grouped.items()}


def unique_energies(group_runs):
    return sorted({float(r["energy"]) for r in group_runs})


def is_trainable_group(group_runs):
    # A regression energy identifier needs at least two known energies.
    return len(unique_energies(group_runs)) >= 2 and len(group_runs) >= 2


def build_group_status_rows(grouped):
    rows = []
    for key in sorted(grouped, key=group_sort_key):
        group_runs = grouped[key]
        energies = unique_energies(group_runs)
        trainable = is_trainable_group(group_runs)
        if trainable:
            status = "trained separately"
            note = "OK: at least two known energies in this particle/bias/reconstruction group"
        else:
            status = "not trained"
            note = "Only one known energy in this group; a real energy calibration curve cannot be learned"

        rows.append({
            "group": group_text(key),
            "particle": key[0],
            "bias_v": float(key[1]),
            "chain": key[2],
            "optical": bool(key[3]),
            "n_runs": len(group_runs),
            "runs": " ".join(str(r["run"]) for r in group_runs),
            "n_unique_energies": len(energies),
            "energies_GeV": " ".join(f"{e:.0f}" for e in energies),
            "status": status,
            "note": note,
        })
    return rows


def train_group_model(group_runs):
    X_train, y_train, _ = stack_events(group_runs)
    model = make_model()
    model.fit(X_train, y_train)
    return model


def leave_one_run_out_validation_grouped(grouped):
    rows = []

    for key in sorted(grouped, key=group_sort_key):
        group_runs = grouped[key]
        if not is_trainable_group(group_runs):
            continue

        allowed_energies = unique_energies(group_runs)

        for test_run in group_runs:
            train_runs = [run for run in group_runs if run["run"] != test_run["run"]]
            train_energies = unique_energies(train_runs)

            if len(train_energies) < 2:
                rows.append({
                    "group": group_text(key),
                    "run": int(test_run["run"]),
                    "particle": test_run["particle"],
                    "bias_v": float(test_run["bias_v"]),
                    "optical": bool(test_run.get("optical", False)),
                    "chain": test_run["chain"],
                    "true_energy": float(test_run["energy"]),
                    "identified_energy": np.nan,
                    "pred_median": np.nan,
                    "pred_mean": np.nan,
                    "pred_spread": np.nan,
                    "abs_error_median": np.nan,
                    "validation_status": "skipped: after leaving this run out, fewer than two energies remain",
                })
                continue

            model = train_group_model(train_runs)
            pred_events = model.predict(test_run["X"])
            pred_median = float(np.median(pred_events))
            pred_mean = float(np.mean(pred_events))
            pred_spread = robust_spread(pred_events)
            identified = nearest_allowed_energy(pred_median, allowed_energies)

            rows.append({
                "group": group_text(key),
                "run": int(test_run["run"]),
                "particle": test_run["particle"],
                "bias_v": int(test_run["bias_v"]),
                "chain": test_run["chain"],
                "true_energy": float(test_run["energy"]),
                "identified_energy": identified,
                "pred_median": pred_median,
                "pred_mean": pred_mean,
                "pred_spread": pred_spread,
                "abs_error_median": abs(pred_median - float(test_run["energy"])),
                "validation_status": "validated",
            })

    return rows


def predict_all_runs_grouped(grouped):
    rows = []

    for key in sorted(grouped, key=group_sort_key):
        group_runs = grouped[key]
        energies = unique_energies(group_runs)
        trainable = is_trainable_group(group_runs)

        if trainable:
            print(f"Training separate model for {group_text(key)}")
            model = train_group_model(group_runs)
        else:
            model = None

        for run in group_runs:
            base = {
                "run": int(run["run"]),
                "particle": run["particle"],
                "bias_v": float(run["bias_v"]),
                "optical": bool(run.get("optical", False)),
                "chain": run["chain"],
                "true_energy": float(run["energy"]),
                "n_events": int(run["X"].shape[0]),
                "model_group": group_text(key),
                "group_unique_energies": len(energies),
                "group_energies_GeV": " ".join(f"{e:.0f}" for e in energies),
            }

            if model is None:
                base.update({
                    "prediction_status": "not predicted: no separate calibration curve for this group",
                    "identified_energy": np.nan,
                    "pred_median": np.nan,
                    "pred_mean": np.nan,
                    "pred_spread": np.nan,
                    "error_median": np.nan,
                    "abs_error_median": np.nan,
                })
            else:
                pred_events = model.predict(run["X"])
                pred_median = float(np.median(pred_events))
                pred_mean = float(np.mean(pred_events))
                pred_spread = robust_spread(pred_events)
                identified = nearest_allowed_energy(pred_median, energies)

                base.update({
                    "prediction_status": "predicted with separate group model",
                    "identified_energy": identified,
                    "pred_median": pred_median,
                    "pred_mean": pred_mean,
                    "pred_spread": pred_spread,
                    "error_median": pred_median - float(run["energy"]),
                    "abs_error_median": abs(pred_median - float(run["energy"])),
                })

            rows.append(base)

    rows.sort(key=lambda r: r["run"])
    return rows

# =========================================================
# CSV OUTPUT
# =========================================================

def clean_for_csv(value):
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean_for_csv(row.get(k, "")) for k in fieldnames})
    print("Saved:", path)


def save_all_csv(rows, loo_rows, group_rows):
    write_csv(
        OUT_RESULTS_CSV,
        rows,
        [
            "run",
            "particle",
            "bias_v",
            "optical",
            "chain",
            "true_energy",
            "prediction_status",
            "identified_energy",
            "pred_median",
            "pred_mean",
            "pred_spread",
            "error_median",
            "abs_error_median",
            "n_events",
            "model_group",
            "group_unique_energies",
            "group_energies_GeV",
        ],
    )

    write_csv(
        OUT_LOO_CSV,
        loo_rows,
        [
            "group",
            "run",
            "particle",
            "bias_v",
            "optical",
            "chain",
            "true_energy",
            "identified_energy",
            "pred_median",
            "pred_mean",
            "pred_spread",
            "abs_error_median",
            "validation_status",
        ],
    )

    write_csv(
        OUT_GROUPS_CSV,
        group_rows,
        [
            "group",
            "particle",
            "bias_v",
            "optical",
            "chain",
            "n_runs",
            "runs",
            "n_unique_energies",
            "energies_GeV",
            "status",
            "note",
        ],
    )

# =========================================================
# PLOT OUTPUT
# =========================================================

def finite_rows(rows, key):
    out = []
    for r in rows:
        v = r.get(key, np.nan)
        try:
            if np.isfinite(float(v)):
                out.append(r)
        except Exception:
            pass
    return out


def plot_grouped_results(rows, loo_rows, group_rows, out_fig):
    """
    Make only the two requested figure elements:
      1) leave-one-run-out validation plot
      2) calibration-group table

    The old per-run hadron/electron panels are intentionally removed.
    """
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    def short_group_label(group_name):
        parts = [x.strip() for x in group_name.split(",")]
        particle = parts[0]
        bias_txt = parts[1]
        chain = parts[2]
        optical = parts[3] if len(parts) > 3 else "no optical"
        bias_val = bias_txt.replace("bias", "").replace("V", "").strip()
        chain_short = "EM" if chain == "EventMatch" else "std"
        optical_short = "/opt" if optical == "optical" else ""
        return f"{particle}/{bias_val}V/{chain_short}{optical_short}"

    def wrap_runs_text(run_string, items_per_line=4):
        parts = str(run_string).split()
        if len(parts) <= items_per_line:
            return str(run_string)
        lines = []
        for i in range(0, len(parts), items_per_line):
            lines.append(" ".join(parts[i:i + items_per_line]))
        return "\n".join(lines)

    fig = plt.figure(figsize=(11.0, 10.8), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.75],
        left=0.08,
        right=0.98,
        bottom=0.06,
        top=0.96,
        hspace=0.34,
    )

    ax_loo = fig.add_subplot(gs[0, 0])
    ax_group = fig.add_subplot(gs[1, 0])
    ax_group.axis("off")

    # -----------------------------------------------------
    # Figure 1: leave-one-run-out validation
    # -----------------------------------------------------
    valid_loo = [r for r in loo_rows if r["validation_status"] == "validated"]

    if valid_loo:
        groups = sorted(set(r["group"] for r in valid_loo))
        marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]

        all_true = []
        all_pred = []
        metric_lines = []

        for gi, group in enumerate(groups):
            g_rows = [r for r in valid_loo if r["group"] == group]
            true = np.array([r["true_energy"] for r in g_rows], dtype=float)
            pred = np.array([r["pred_median"] for r in g_rows], dtype=float)
            all_true.extend(true.tolist())
            all_pred.extend(pred.tolist())

            ax_loo.scatter(
                true,
                pred,
                s=42,
                marker=marker_cycle[gi % len(marker_cycle)],
                label=short_group_label(group),
                zorder=3,
            )

            mae = mean_absolute_error(true, pred)
            rmse = np.sqrt(mean_squared_error(true, pred))
            metric_lines.append(
                f"{short_group_label(group)}: MAE {mae:.1f}, RMSE {rmse:.1f} GeV"
            )

        lo = min(all_true + all_pred) - 20
        hi = max(all_true + all_pred) + 20
        ax_loo.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="0.35", label="perfect")
        ax_loo.set_xlim(lo, hi)
        ax_loo.set_ylim(lo, hi)
        ax_loo.text(
            0.03,
            0.97,
            "Leave-one-run-out\n" + "\n".join(metric_lines),
            transform=ax_loo.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="0.97", edgecolor="0.72"),
        )
    else:
        ax_loo.text(
            0.5,
            0.5,
            "No group has enough calibration data\nfor leave-one-run-out validation.",
            ha="center",
            va="center",
            transform=ax_loo.transAxes,
        )

    ax_loo.set_title("Run-wise validation within separate groups", fontweight="bold")
    ax_loo.set_xlabel("True beam energy [GeV]")
    ax_loo.set_ylabel("Predicted energy [GeV]")
    ax_loo.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax_loo.legend(loc="lower right", frameon=True, ncol=2, fontsize=6.9)

    # -----------------------------------------------------
    # Figure 2: calibration-group table
    # -----------------------------------------------------
    table_rows = []
    for gr in group_rows:
        group_name = f"{gr['particle']}, {bias_to_text(gr['bias_v'])} V, {gr['chain']}"
        if gr.get("optical", False):
            group_name += ", optical"
        table_rows.append([
            group_name,
            str(gr["n_runs"]),
            wrap_runs_text(gr["runs"], items_per_line=4),
            wrap_runs_text(gr["energies_GeV"], items_per_line=4),
            gr["status"],
        ])

    table = ax_group.table(
        cellText=table_rows,
        colLabels=["Group", "N", "Runs", "E [GeV]", "Status"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.22, 0.08, 0.31, 0.17, 0.22],
        bbox=[0.00, 0.00, 1.00, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_edgecolor("0.75")
        cell.get_text().set_wrap(True)
        if row == 0:
            cell.set_facecolor("0.84")
            cell.set_text_props(weight="bold")
        else:
            status = group_rows[row - 1]["status"]
            cell.set_facecolor("0.93" if status == "trained separately" else "0.98")
            if col in (2, 3):
                cell.set_text_props(fontsize=6.5)

    ax_group.set_title("Calibration groups used by the corrected method", fontweight="bold", pad=10)

    fig.savefig(out_fig, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_fig)

# =========================================================
# MAIN
# =========================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = load_all_runs()
    grouped = split_runs_by_group(runs)
    group_rows = build_group_status_rows(grouped)

    print("\nCalibration groups:")
    for row in group_rows:
        print(
            f"- {row['group']}: runs [{row['runs']}], energies [{row['energies_GeV']}], {row['status']}"
        )

    print("\nRunning leave-one-run-out validation inside each trainable group...")
    loo_rows = leave_one_run_out_validation_grouped(grouped)

    print("\nTraining separate final models and predicting only within matching groups...")
    rows = predict_all_runs_grouped(grouped)

    save_all_csv(rows, loo_rows, group_rows)
    plot_grouped_results(rows, loo_rows, group_rows, OUT_FIG)

    print("\nDone.")
    print("Main figure:", OUT_FIG)
    print("Results CSV:", OUT_RESULTS_CSV)
    print("Validation CSV:", OUT_LOO_CSV)
    print("Group status CSV:", OUT_GROUPS_CSV)


if __name__ == "__main__":
    main()
