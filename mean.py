import re
import glob
import statistics as stats
import csv

# Point this to your log files (3 runs)
LOG_FILES = [
    "/home/cc/expr/dlrm_minrui/prime_merge_split_multibatch_runs/run1_batch8190/run1_batch8190_summary.txt",
    "/home/cc/expr/dlrm_minrui/prime_merge_split_multibatch_runs/run2_batch8190/run2_batch8190_summary.txt",
    "/home/cc/expr/dlrm_minrui/prime_merge_split_multibatch_runs/run3_batch8190/run3_batch8190_summary.txt"
]

config_re = re.compile(
    r"Configuration: Threads=(\d+), Merge=(\d+), Split=(\d+), NumSplits=(\d+)"
)
float_re = re.compile(r"([-+]?\d*\.\d+|\d+)")


def parse_run(filepath):
    results = []
    with open(filepath, "r") as f:
        text = f.read()

    # Split by configuration blocks
    blocks = text.split("Configuration: ")
    for block in blocks:
        block = block.strip()
        if not block.startswith("Threads="):
            continue

        header_line, *rest = block.splitlines()
        m = config_re.match("Configuration: " + header_line)
        if not m:
            # try without the "Configuration: "
            m = config_re.match(header_line)
        if not m:
            continue

        threads, merge, split, numsplits = map(int, m.groups())
        block_text = "\n".join(rest)
        
        def extract_one(pattern):
            m = re.search(pattern, block_text)
            if not m:
                return None
            
            # Extract the value after '='
            matched_str = m.group(0)
            if '=' in matched_str:
                value_part = matched_str.split('=', 1)[1]
                fm = float_re.search(value_part)
                return float(fm.group(0)) if fm else None
            
            # Fallback for patterns without '='
            fm = float_re.search(matched_str)
            return float(fm.group(0)) if fm else None

        peak = extract_one(r"PEAK_GBps=.*")
        avg = extract_one(r"AVG_GBps=.*")
        l1 = extract_one(r"L1_MISS_RATE=.*")
        llc = extract_one(r"LLC_MISS_RATE=.*")
        mlp = extract_one(r"The MLP time is .*")
        emb = extract_one(r"The embedding time is .*")
        inter = extract_one(r"The interaction time is .*")
        total = extract_one(r"The total time is .*")

        results.append({
            "Threads": threads,
            "Merge": merge,
            "Split": split,
            "NumSplits": numsplits,
            "PEAK_GBps": peak,
            "AVG_GBps": avg,
            "L1_MISS_RATE": l1,
            "LLC_MISS_RATE": llc,
            "MLP_time": mlp,
            "Embedding_time": emb,
            "Interaction_time": inter,
            "Total_time": total,
        })

    return results


# Aggregate all runs
from collections import defaultdict

data_by_config = defaultdict(lambda: {k: [] for k in [
    "PEAK_GBps", "AVG_GBps", "L1_MISS_RATE", "LLC_MISS_RATE",
    "MLP_time", "Embedding_time", "Interaction_time", "Total_time"
]})

for path in LOG_FILES:
    for rec in parse_run(path):
        key = (rec["Threads"], rec["Merge"], rec["Split"], rec["NumSplits"])
        for metric in data_by_config[key].keys():
            if rec[metric] is not None:
                data_by_config[key][metric].append(rec[metric])

# Compute means and write table
out_file = "dlrm_split_means_bs8190.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Threads", "Merge", "Split", "NumSplits",
        "PEAK_GBps_mean", "AVG_GBps_mean",
        "L1_MISS_RATE_mean", "LLC_MISS_RATE_mean",
        "MLP_time_mean", "Embedding_time_mean",
        "Interaction_time_mean", "Total_time_mean",
        "Num_runs"
    ])

    for (threads, merge, split, numsplits), metrics in sorted(data_by_config.items()):
        counts = {m: len(v) for m, v in metrics.items()}
        # Number of runs that actually had that config
        num_runs = max(counts.values()) if counts else 0

        def m_or_none(vals):
            return stats.mean(vals) if vals else None

        writer.writerow([
            threads, merge, split, numsplits,
            m_or_none(metrics["PEAK_GBps"]),
            m_or_none(metrics["AVG_GBps"]),
            m_or_none(metrics["L1_MISS_RATE"]),
            m_or_none(metrics["LLC_MISS_RATE"]),
            m_or_none(metrics["MLP_time"]),
            m_or_none(metrics["Embedding_time"]),
            m_or_none(metrics["Interaction_time"]),
            m_or_none(metrics["Total_time"]),
            num_runs,
        ])

print(f"Wrote mean table to {out_file}")
