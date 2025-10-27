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
            'pid': ev.get('pid'),
            'args': ev.get('args', {})
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

def decode_duration_distribution(pattern: str = 'llm: decode', bins: int = 50) -> None:
    """Produce a combined chart with:
    1) bar chart: count of 'llm: decode' events for each token length
    2) line chart: throughput (total_tokens / total_time) per token length

    Both use token length as the x-axis. The bar chart uses the left y-axis
    (counts) and the throughput line uses the right y-axis (tokens/sec).
    """
    pat = pattern.lower()
    # gather (duration_s, tokens) for matching events
    pairs = []
    for r in records:
        name = r.get('name')
        dur = r.get('dur')
        if not (isinstance(name, str) and name.lower().startswith(pat) and dur is not None):
            continue
        try:
            d_s = float(dur) / 1_000_000.0
        except Exception:
            continue
        args = r.get('args') or {}
        tok = args.get('tokens')
        if tok is None:
            tok = args.get('n_tokens') or args.get('token_count')
        try:
            tok_i = int(float(tok)) if tok is not None else None
        except Exception:
            tok_i = None
        if tok_i is None:
            continue
        pairs.append((tok_i, d_s))

    out_dir = Path('inference/profiler_traces')
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pairs:
        print(f"No 'llm: decode' events with token counts found for pattern '{pattern}'.")
        return

    # aggregate by token length
    agg_counts = collections.Counter()
    agg_total_time: Dict[int, float] = collections.defaultdict(float)
    agg_total_tokens: Dict[int, float] = collections.defaultdict(float)
    for tok_i, dur_s in pairs:
        agg_counts[tok_i] += 1
        agg_total_time[tok_i] += float(dur_s)
        agg_total_tokens[tok_i] += float(tok_i)

    lengths = sorted(agg_counts.keys())
    counts = np.array([agg_counts[L] for L in lengths], dtype=int)
    total_times = np.array([agg_total_time[L] for L in lengths], dtype=float)
    total_tokens = np.array([agg_total_tokens[L] for L in lengths], dtype=float)

    # throughput defined as total_tokens / total_time (tokens per second)
    # avoid division by zero
    total_times_safe = np.where(total_times == 0.0, 1e-9, total_times)
    throughput = total_tokens / total_times_safe

    # Build the combined plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.array(lengths)
    # bar: counts on left y-axis
    ax.bar(x, counts, color='steelblue', alpha=0.9, label='count')
    ax.set_xlabel('Decode token length')
    ax.set_ylabel('Count')
    ax.set_title(f"'llm: decode' counts and throughput by token length (n={len(pairs)})")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{int(v)}"))

    # line: throughput on right y-axis
    ax2 = ax.twinx()
    ax2.plot(x, throughput, '-o', color='darkorange', label='throughput (tokens/s)')
    ax2.set_ylabel('Throughput (tokens/sec)')
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v:.2f}"))

    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='best', fontsize=8)

    plt.tight_layout()
    out_png = out_dir / 'llm_decode_counts_and_throughput.png'
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # Save CSV/JSON summary
    import json as _json
    rows = ['length,count,total_time_s,total_tokens,throughput_tps']
    for L, c, tt, to, thr in zip(lengths, counts.tolist(), total_times.tolist(), total_tokens.tolist(), throughput.tolist()):
        rows.append(f"{L},{c},{tt:.9f},{to:.6f},{thr:.6f}")
    (out_dir / 'llm_decode_counts_and_throughput.csv').write_text('\n'.join(rows), encoding='utf-8')
    (out_dir / 'llm_decode_counts_and_throughput.json').write_text(_json.dumps({'summary': {'total_samples': len(pairs)}, 'data': [
        {'length': int(L), 'count': int(c), 'total_time_s': float(tt), 'total_tokens': float(to), 'throughput_tps': float(thr)}
        for L, c, tt, to, thr in zip(lengths, counts.tolist(), total_times.tolist(), total_tokens.tolist(), throughput.tolist())
    ]}, indent=2), encoding='utf-8')

    print(f"Saved combined counts+throughput plot to {out_png} and CSV/JSON summaries.")

def tokens_share_by_tid() -> None:
    """For each tid, compute the total tokens per event name and plot their share (stacked percentage bars).

    Uses record args['tokens'] when present; missing tokens are treated as 0.
    Saves PNG to inference/profiler_traces/tokens_share_by_tid.png and prints average share per name.
    """
    # Group events by tid
    by_tid = collections.defaultdict(list)
    for r in records:
        by_tid[r.get('tid')].append(r)

    # Sum tokens per name for each tid
    result = {}
    for tid, items in by_tid.items():
        sums: Dict[str, float] = collections.defaultdict(float)
        for it in items:
            name = it.get('name')
            args = it.get('args') or {}
            tok = args.get('tokens', 0)
            try:
                tok_f = float(tok)
            except Exception:
                tok_f = 0.0
            if name is None:
                continue
            sums[name] += tok_f
        result[tid] = dict(sums)

    if not result:
        print("No token data found across tids.")
        return

    tids = sorted(list(result.keys()), key=lambda x: int(x) if str(x).isdigit() else str(x))
    names = sorted({n for d in result.values() for n in d.keys()})

    # Build absolute token matrix: rows=names, cols=tids
    mat = np.zeros((len(names), len(tids)), dtype=float)
    for j, tid in enumerate(tids):
        for i, name in enumerate(names):
            mat[i, j] = float(result[tid].get(name, 0.0))

    # Normalize per tid to get shares (avoid division by zero)
    col_sums = mat.sum(axis=0)
    denom = np.where(col_sums == 0.0, 1.0, col_sums)
    share = mat / denom

    # Plot stacked percentage bars
    fig, ax = plt.subplots(figsize=(max(6, len(tids) * 0.6), 6))
    bottoms = np.zeros(len(tids), dtype=float)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / max(1, len(names) - 1)) for i in range(len(names))]
    x = np.arange(len(tids))
    # draw stacked bars using numeric x positions and set xticklabels later
    for i, name in enumerate(names):
        vals = share[i, :]
        bars = ax.bar(x, vals, bottom=bottoms, label=name, color=colors[i % len(colors)])
        # add numeric token labels inside the bar segments when meaningful
        for j in range(len(tids)):
            tok_count = mat[i, j]
            seg_share = vals[j]
            # show label when tokens >=1 or share >= 2%
            if tok_count >= 1 or seg_share >= 0.02:
                # center y position of this segment
                y = bottoms[j] + seg_share / 2.0
                # label text
                txt = f"{int(tok_count):d}"
                # choose text color for contrast
                r, g, b, a = colors[i % len(colors)]
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = 'white' if luminance < 0.5 else 'black'
                ax.text(x[j], y, txt, ha='center', va='center', fontsize=7, color=text_color)
        bottoms += vals
    # apply tid labels
    ax.set_xticks(x)
    ax.set_xticklabels(tids, rotation=45, ha='right')

    ax.set_ylabel('Percentage')
    ax.set_title('KV Cache distribution for each task')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    plt.tight_layout()

    out_dir = Path('inference/profiler_traces')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'tokens_share_by_tid.png'
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    # Also report the average share per event name across tids (including zeros)
    avg_share_per_name = {name: float(share[i, :].mean()) for i, name in enumerate(names)}
    print("Average token share per name across all tids:")
    total_share = sum(avg_share_per_name.values()) or 1.0
    for name, val in sorted(avg_share_per_name.items(), key=lambda x: x[1], reverse=True):
        perc = val * 100.0
        print(f"  {name}: {perc:.2f}%")

def concurrent_events_over_time(bin_ms: int = 100) -> None:
    """统计所有事件类型（name），每个事件按区间(ts, ts+dur)统计并发数，全部画在同一张总览图。"""
    # 收集所有事件区间
    intervals = collections.defaultdict(list)  # name -> list of (start, end)
    for r in records:
        name = r.get('name')
        ts = r.get('ts')
        dur = r.get('dur')
        if not (isinstance(name, str) and ts is not None and dur is not None):
            continue
        start = float(ts)
        end = start + float(dur)
        intervals[name].append((start, end))

    # 统一时间轴
    all_times = [t for ivs in intervals.values() for t in sum(ivs, ())]
    if not all_times:
        print("No events found for concurrency analysis.")
        return
    t_min = min(all_times)
    t_max = max(all_times)
    bin_us = bin_ms * 1000
    bins = np.arange(t_min, t_max + bin_us, bin_us)
    bin_centers = bins[:-1] + bin_us / 2

    # 对每种事件，统计每个时间 bin 内正在运行的数量
    lines = {}
    for name, ivs in intervals.items():
        counts = np.zeros(len(bins) - 1, dtype=int)
        for start, end in ivs:
            idx_start = np.searchsorted(bins, start, side='right') - 1
            idx_end = np.searchsorted(bins, end, side='left') - 1
            for idx in range(idx_start, min(idx_end + 1, len(counts))):
                counts[idx] += 1
        lines[name] = counts

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, counts) in enumerate(lines.items()):
        ax.plot(bin_centers / 1e6, counts, label=name, linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Concurrent running count')
    ax.set_title('Concurrent events over time (all names)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    plt.tight_layout()
    out_dir = Path('inference/profiler_traces')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'concurrent_events_over_time.png'
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved all-names concurrent events chart to {out_file}")

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
    # Also produce the 'llm: decode' duration distribution
    decode_duration_distribution("llm: decode")
    print("Saved llm: decode duration distribution artifacts under inference/profiler_traces/")
    # Produce token share by tid stacked percentage chart
    tokens_share_by_tid()
    print("Saved tokens share by tid chart under inference/profiler_traces/")
    # Analyze and plot concurrency of events over time
    concurrent_events_over_time()
    print("Saved concurrent events over time chart under inference/profiler_traces/")
