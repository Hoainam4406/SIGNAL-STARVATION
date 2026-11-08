import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd


def aggregate_results(
    results_dir: Path,
    experiment_name: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Tìm và gom toàn bộ các file eval_results.csv theo cấu trúc:
    results/<experiment_name>/<timestamp>/eval_results.csv
    Sau đó tính toán các thống kê (mean, std, min, max, count) qua các seed.
    """
    if not results_dir.exists():
        print(f"[ERROR] Results directory does not exist: {results_dir}")
        return None

    # Tìm tất cả file eval_results.csv
    pattern = "**/*eval_results.csv"
    if experiment_name:
        pattern = f"{experiment_name}/**/eval_results.csv"

    csv_files: List[Path] = list(results_dir.glob(pattern))

    if not csv_files:
        print(f"[WARNING] No 'eval_results.csv' files found in {results_dir} (pattern: {pattern})")
        return None

    print(f"[*] Found {len(csv_files)} evaluation result file(s):")
    dfs: List[pd.DataFrame] = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
                print(f"  - Loaded: {f.relative_to(results_dir)} ({len(df)} rows)")
        except Exception as e:
            print(f"  [!] Error reading {f}: {e}")

    if not dfs:
        print("[WARNING] All found CSV files were empty or corrupted.")
        return None

    all_data = pd.concat(dfs, ignore_index=True)

    # Các cột định lượng cần tổng hợp thống kê
    metric_cols = [
        "total_reward",
        "success",
        "steps_to_goal",
        "collision_count",
        "spl",
        "mean_progress",
        "sa_uniform_mean",
        "training_time_s",
    ]
    # Lọc chỉ các metric thực sự có trong dữ liệu
    available_metrics = [c for c in metric_cols if c in all_data.columns]

    # Các cột phân nhóm (gom theo thuật toán, môi trường, reward, v.v.)
    group_cols = ["algorithm", "dataset", "robot", "reward_type", "level_filter"]
    available_groups = [c for c in group_cols if c in all_data.columns]

    print("\n" + "=" * 80)
    print("AGGREGATED BENCHMARK SUMMARY (ACROSS ALL SEEDS)")
    print("=" * 80)

    # Thống kê tổng hợp: mean và std
    summary = (
        all_data.groupby(available_groups)[available_metrics]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )

    # Tạo bảng hiển thị tóm tắt đẹp mắt (mean ± std)
    display_rows = []
    for name, group in all_data.groupby(available_groups):
        if not isinstance(name, tuple):
            name = (name,)
        row_dict = dict(zip(available_groups, name))
        row_dict["n_samples"] = len(group)
        row_dict["seeds"] = ",".join(map(str, sorted(group["seed"].unique()))) if "seed" in group else "N/A"

        for m in available_metrics:
            m_mean = group[m].mean()
            m_std = group[m].std()
            if pd.isna(m_std):
                m_std = 0.0
            row_dict[f"{m}_mean_std"] = f"{m_mean:.3f} ± {m_std:.3f}"

        display_rows.append(row_dict)

    summary_df = pd.DataFrame(display_rows)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Lưu kết quả tổng hợp ra file CSV
    if output_path is None:
        output_path = results_dir / "aggregated_summary.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[+] Aggregated summary saved successfully to: {output_path}")

    # Đồng thời lưu bản chi tiết đầy đủ (multi-index stats)
    detailed_path = output_path.parent / "aggregated_detailed_stats.csv"
    summary.to_csv(detailed_path, encoding="utf-8")
    print(f"[+] Detailed multi-stat table saved to: {detailed_path}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate experimental evaluation results across seeds.")
    project_root = Path(__file__).resolve().parent.parent
    default_results_dir = project_root / "results"

    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(default_results_dir),
        help="Path to results directory.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Filter by experiment name subdirectory.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path for output aggregated summary CSV.",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output_file) if args.output_file else None

    aggregate_results(
        results_dir=results_dir,
        experiment_name=args.experiment_name,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
