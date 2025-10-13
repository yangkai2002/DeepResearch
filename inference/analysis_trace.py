from pathlib import Path
import json
import argparse
from typing import List, Dict, Any
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Paper-style plotting defaults: serif font, compact sizes, high DPI
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIX', 'Palatino'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': False,
})

records: List[Dict[str, Any]] = []

def _short_tid(tid_value: Any) -> str:
    """
    Return the simplified tid: the second-to-last underscore-separated token
    for strings like 'ThreadPoolExecutor-0_1_140303926138432' return '1'.
    """
    s = str(tid_value)
    parts = s.split('_')
    return parts[-2]

def load_trace(path: Path) -> None:
    """Load trace file and populate global records."""
    global records
    events = json.loads(path.read_text(encoding='utf-8'))['traceEvents']
    out: List[Dict[str, Any]] = []
    for ev in events:
        out.append({
            'name': ev.get('name'),
            'ts': ev.get('ts'),
            'dur': ev.get('dur'),
            'tid': _short_tid(ev.get('tid')),
            'pid': ev.get('pid')
        })
    records = out

def task_time_for_each_tid() -> None:
    """Group `records` by tid and sum durations per event name."""

    by_tid = collections.defaultdict(list)
    for r in records:
        by_tid[r.get('tid')].append(r)

    result = {}
    for tid, items in by_tid.items():
        sums: Dict[str, float] = collections.defaultdict(float)
        for it in items:
            name = it['name']
            sums[name] += float(it['dur'])
        result[tid] = dict(sums)
        
    tids = sorted(list(result.keys()), key=lambda x: int(x) if x.isdigit() else x)
    names = sorted({n for d in result.values() for n in d.keys()})
    mat = np.zeros((len(names), len(tids)), dtype=float)
    for j, tid in enumerate(tids):
        for i, name in enumerate(names):
            mat[i, j] = result[tid].get(name, 0.0) / 1_000_000.0
    
    fig, ax = plt.subplots(figsize=(max(6, len(tids) * 0.6), 6))
    bottoms = np.zeros(len(tids), dtype=float)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / max(1, len(names) - 1)) for i in range(len(names))]
    for i, name in enumerate(names):
        ax.bar(tids, mat[i, :], bottom=bottoms, label=name, color=colors[i % len(colors)])
        bottoms += mat[i, :]

    ax.set_ylabel('Total duration (s)')
    ax.set_title('Total time per event name by tid')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v:.2f}"))
    plt.tight_layout()
    out_dir = Path('inference/profiler_traces')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'tasks_by_tid.png'
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    # Also report the average time per event name across all tids (including zeros) and percentage
    avg_per_name = {name: float(mat[i, :].mean()) for i, name in enumerate(names)}
    # Print sorted by descending average seconds
    print("Average time per name across all tids (seconds):")
    sum_total = sum(avg_per_name.values())
    for name, val in sorted(avg_per_name.items(), key=lambda x: x[1], reverse=True):
        perc = (val / sum_total * 100) if sum_total > 0 else 0.0
        print(f"  {name}: {val:.6f}s {perc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a Chrome-style trace into memory')
    parser.add_argument('path', type=str, help='Trace JSON file to load')
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f"Trace file not found: {p}")

    load_trace(p)
    print(f"Loaded {len(records)} events from {p}")
    
    task_time_for_each_tid()
    print(f"Saved to inference/profiler_traces/tasks_by_tid.png")
