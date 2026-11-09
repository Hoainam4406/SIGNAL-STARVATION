"""Map Profiler & Topological Complexity Analysis for NavSim Grid Environments.

Extracts geometric and topological features from grid JSON files:
- free_cells, main_cells, main_frac, W, H, cell_m
- Obstacle density: 1 - main_frac
- Navigable area (m²): main_cells * cell_m² (main_m2)
- Map complexity classification: Small / Medium / Large based on main_m2
- Task difficulty classification: Easy / Medium / Hard / Starved based on sa_uniform
"""

import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import navsim


class MapProfiler:
    """Profiler for analyzing map topological complexity and task difficulty in NavSim datasets."""

    def __init__(
        self,
        dataset_name: str = "grid-turtlebot4-v2",
        robot_name: str = "turtlebot4",
        datasets_dir: Optional[Union[str, Path]] = None,
    ):
        self.dataset_name = dataset_name
        self.robot_name = robot_name
        self.dataset_root = self._locate_dataset(dataset_name, datasets_dir)
        self.grids_dir = self.dataset_root / "grids"
        self.tasks_dir = self.dataset_root / "tasks" / self.robot_name

        if not self.grids_dir.exists():
            raise FileNotFoundError(f"Grids directory not found: {self.grids_dir}")

    def _locate_dataset(
        self,
        dataset_name: str,
        custom_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Locate dataset directory from custom path, navsim registry, or environment variable."""
        if custom_dir:
            p = Path(custom_dir)
            if (p / dataset_name).exists():
                return p / dataset_name
            if p.exists():
                return p

        # Check navsim list_datasets
        try:
            for ds in navsim.list_datasets():
                if ds.name == dataset_name and hasattr(ds, "root") and Path(ds.root).exists():
                    return Path(ds.root)
        except Exception:
            pass

        # Check environment variable NAVSIM_DATASETS
        env_datasets = os.environ.get("NAVSIM_DATASETS")
        if env_datasets:
            p = Path(env_datasets) / dataset_name
            if p.exists():
                return p

        # Check ~/.navsim/datasets
        user_home_ds = Path.home() / ".navsim" / "datasets" / dataset_name
        if user_home_ds.exists():
            return user_home_ds

        # Fallback to local drive paths
        for drive in ["F:/navsim/datasets", "C:/navsim/datasets"]:
            fallback = Path(drive) / dataset_name
            if fallback.exists():
                return fallback

        raise FileNotFoundError(
            f"Could not locate dataset '{dataset_name}'. "
            f"Please specify datasets_dir or set NAVSIM_DATASETS environment variable."
        )

    def load_manifest_splits(self) -> Dict[str, str]:
        """Read manifest.json to map scene_id -> split (train/val/test)."""
        manifest_path = self.dataset_root / "manifest.json"
        map_split_map: Dict[str, str] = {}
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            splits = manifest.get("splits", {})
            for split_name, scene_list in splits.items():
                for scene_id in scene_list:
                    map_split_map[scene_id] = split_name
        return map_split_map

    def profile_maps(self) -> pd.DataFrame:
        """Read all grid JSON files and calculate topological complexity metrics."""
        grid_files = sorted(self.grids_dir.glob("*.json"))
        if not grid_files:
            raise FileNotFoundError(f"No grid JSON files found in {self.grids_dir}")

        scene_splits = self.load_manifest_splits()
        records: List[Dict] = []

        for fpath in grid_files:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            map_id = data.get("map_id", fpath.stem)
            scene = data.get("scene", map_id.split("_")[0])
            storey = data.get("storey", 1)
            w = int(data.get("W", 0))
            h = int(data.get("H", 0))
            cell_m = float(data.get("cell_m", 0.0))
            free_cells = int(data.get("free_cells", 0))
            main_cells = int(data.get("main_cells", 0))
            main_frac = float(data.get("main_frac", 0.0))
            main_m2 = float(data.get("main_m2", main_cells * (cell_m ** 2)))

            # Computed metrics
            obstacle_density = float(1.0 - main_frac)
            navigable_area_m2 = round(main_cells * (cell_m ** 2), 2)
            total_cells = w * h
            split = scene_splits.get(scene, data.get("split", "train"))

            records.append({
                "map_id": map_id,
                "scene": scene,
                "storey": storey,
                "split": split,
                "W": w,
                "H": h,
                "total_cells": total_cells,
                "cell_m": cell_m,
                "free_cells": free_cells,
                "main_cells": main_cells,
                "main_frac": round(main_frac, 6),
                "obstacle_density": round(obstacle_density, 6),
                "main_m2": round(main_m2, 2),
                "navigable_area_m2": navigable_area_m2,
            })

        df = pd.DataFrame(records)

        # Classification based on tertiles (33.3% / 66.7%)
        p33 = float(np.percentile(df["main_m2"], 33.333))
        p67 = float(np.percentile(df["main_m2"], 66.667))

        def classify_tertile(val: float) -> str:
            if val < p33:
                return "Small"
            elif val < p67:
                return "Medium"
            return "Large"

        # Classification based on natural round cutoffs (< 30, 30-50, >= 50 m²)
        def classify_round(val: float) -> str:
            if val < 30.0:
                return "Small"
            elif val < 50.0:
                return "Medium"
            return "Large"

        df["map_complexity"] = df["main_m2"].apply(classify_tertile)
        df["map_complexity_round"] = df["main_m2"].apply(classify_round)
        df["tertile_p33"] = round(p33, 2)
        df["tertile_p67"] = round(p67, 2)

        return df

    def profile_tasks(self) -> pd.DataFrame:
        """Extract all tasks across maps to inspect task-level signal starvation & difficulty."""
        if not self.tasks_dir.exists():
            return pd.DataFrame()

        task_files = sorted(self.tasks_dir.glob("*.json"))
        records: List[Dict] = []

        for fpath in task_files:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            map_id = data.get("map_id", fpath.stem)
            split = data.get("split", "train")
            tasks = data.get("tasks", [])

            for t in tasks:
                labels = t.get("labels", {})
                d_bfs = labels.get("d_bfs", 0)
                sa_uniform = labels.get("sa_uniform", 0.0)
                sa_persist = labels.get("sa_persist", 0.0)
                starved = bool(labels.get("starved", False))
                level = labels.get("level", "unknown")
                dist_bin = labels.get("dist_bin", "unknown")

                records.append({
                    "map_id": map_id,
                    "split": split,
                    "task_id": t.get("id"),
                    "geodesic_m": t.get("geodesic_m"),
                    "euclidean_m": t.get("euclidean_m"),
                    "d_bfs": d_bfs,
                    "sa_uniform": sa_uniform,
                    "sa_persist": sa_persist,
                    "starved": starved,
                    "level": level,
                    "dist_bin": dist_bin,
                })

        return pd.DataFrame(records)

    def export_csv(
        self,
        df_maps: pd.DataFrame,
        output_path: Union[str, Path] = "audit_reports/map_complexity_profile.csv",
    ) -> Path:
        """Save map complexity profile to CSV."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_maps.to_csv(out, index=False, encoding="utf-8")
        return out

    def export_task_csv(
        self,
        df_tasks: pd.DataFrame,
        output_path: Union[str, Path] = "audit_reports/task_complexity_profile.csv",
    ) -> Path:
        """Save task complexity profile to CSV."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_tasks.to_csv(out, index=False, encoding="utf-8")
        return out

    def plot_distributions(
        self,
        df_maps: pd.DataFrame,
        df_tasks: Optional[pd.DataFrame] = None,
        output_dir: Union[str, Path] = "audit_reports",
    ) -> List[Path]:
        """Generate distribution plots for maps and tasks."""
        out_dir = Path(output_dir)
        logs_dir = out_dir / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        generated_plots: List[Path] = []
        sns.set_theme(style="whitegrid", font="sans-serif")

        # -------------------------------------------------------------
        # 1. Map Distributions (main_m2, main_frac, free_cells, obstacle_density)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) Navigable Area (main_m2) with cutoffs
        ax0 = axes[0, 0]
        sns.histplot(df_maps["main_m2"], bins=25, kde=True, ax=ax0, color="#1f77b4")
        p33 = float(df_maps["tertile_p33"].iloc[0])
        p67 = float(df_maps["tertile_p67"].iloc[0])
        ax0.axvline(p33, color="orange", linestyle="--", linewidth=2, label=f"P33: {p33:.1f} m² (Small)")
        ax0.axvline(p67, color="red", linestyle="--", linewidth=2, label=f"P67: {p67:.1f} m² (Large)")
        ax0.set_title(r"Distribution of Navigable Area ($main\_m^2$)", fontsize=13, fontweight="bold")
        ax0.set_xlabel(r"Navigable Area ($m^2$)", fontsize=11)
        ax0.set_ylabel("Number of Maps", fontsize=11)
        ax0.legend(frameon=True)

        # (b) Obstacle Density (1 - main_frac)
        ax1 = axes[0, 1]
        sns.histplot(df_maps["obstacle_density"], bins=20, kde=True, ax=ax1, color="#d62728")
        ax1.set_title(r"Distribution of Obstacle Density ($1 - main\_frac$)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Obstacle Density (ratio)", fontsize=11)
        ax1.set_ylabel("Number of Maps", fontsize=11)

        # (c) Main Fraction (main_frac)
        ax2 = axes[1, 0]
        sns.histplot(df_maps["main_frac"], bins=20, kde=True, ax=ax2, color="#2ca02c")
        ax2.set_title(r"Distribution of Navigable Fraction ($main\_frac$)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Main Component Fraction", fontsize=11)
        ax2.set_ylabel("Number of Maps", fontsize=11)

        # (d) Free Cells vs Main Cells
        ax3 = axes[1, 1]
        sns.scatterplot(
            data=df_maps,
            x="free_cells",
            y="main_cells",
            hue="map_complexity",
            palette={"Small": "#2ca02c", "Medium": "#ff7f0e", "Large": "#d62728"},
            alpha=0.8,
            s=60,
            ax=ax3,
        )
        ax3.set_title("Free Cells vs Main Component Cells", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Free Cells", fontsize=11)
        ax3.set_ylabel("Main Navigable Cells", fontsize=11)
        ax3.legend(title="Map Scale", frameon=True)

        plt.tight_layout()

        map_plot_path1 = out_dir / "map_distribution.png"
        map_plot_path2 = logs_dir / "map_complexity_distribution.png"
        fig.savefig(map_plot_path1, dpi=300)
        fig.savefig(map_plot_path2, dpi=300)
        plt.close(fig)

        generated_plots.extend([map_plot_path1, map_plot_path2])

        # -------------------------------------------------------------
        # 2. Task Difficulty Distribution (sa_uniform, level, d_bfs)
        # -------------------------------------------------------------
        if df_tasks is not None and not df_tasks.empty:
            fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

            palette = {"easy": "#2ca02c", "medium": "#1f77b4", "hard": "#ff7f0e", "starved": "#d62728"}

            # (a) Task Counts by Level
            ax2_0 = axes2[0, 0]
            sns.countplot(
                data=df_tasks,
                x="level",
                hue="level",
                order=["easy", "medium", "hard", "starved"],
                palette=palette,
                legend=False,
                ax=ax2_0,
            )
            for p in ax2_0.patches:
                val = int(p.get_height())
                pct = (val / len(df_tasks)) * 100
                ax2_0.annotate(
                    f"{val} ({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2.0, val),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    xytext=(0, 3),
                    textcoords="offset points",
                )
            ax2_0.set_title("Task Distribution by Difficulty Level", fontsize=13, fontweight="bold")
            ax2_0.set_xlabel("Task Level", fontsize=11)
            ax2_0.set_ylabel("Number of Tasks", fontsize=11)

            # (b) sa_uniform (log scale) by Level
            ax2_1 = axes2[0, 1]
            df_tasks_plot = df_tasks.copy()
            df_tasks_plot["log_sa_uniform"] = np.log10(np.clip(df_tasks_plot["sa_uniform"], 1e-12, 1.0))
            sns.boxplot(
                data=df_tasks_plot,
                x="level",
                y="log_sa_uniform",
                hue="level",
                order=["easy", "medium", "hard", "starved"],
                palette=palette,
                legend=False,
                ax=ax2_1,
            )
            ax2_1.set_title(r"Signal Density: $\log_{10}(sa\_uniform)$ by Level", fontsize=13, fontweight="bold")
            ax2_1.set_xlabel("Task Level", fontsize=11)
            ax2_1.set_ylabel(r"$\log_{10}(P(\text{reach goal} \mid \text{random}))$", fontsize=11)

            # (c) BFS Path Length (d_bfs) by Level
            ax2_2 = axes2[1, 0]
            sns.boxplot(
                data=df_tasks,
                x="level",
                y="d_bfs",
                hue="level",
                order=["easy", "medium", "hard", "starved"],
                palette=palette,
                legend=False,
                ax=ax2_2,
            )
            ax2_2.set_title(r"Shortest Path Length ($d_{bfs}$ cells) by Level", fontsize=13, fontweight="bold")
            ax2_2.set_xlabel("Task Level", fontsize=11)
            ax2_2.set_ylabel("BFS Distance (cells)", fontsize=11)

            # (d) Distance Bins breakdown
            ax2_3 = axes2[1, 1]
            dist_order = ["1-3", "3-6", "6-10", "10-15"]
            sns.countplot(
                data=df_tasks,
                x="dist_bin",
                hue="level",
                order=[b for b in dist_order if b in df_tasks["dist_bin"].values],
                hue_order=["easy", "medium", "hard", "starved"],
                palette=palette,
                ax=ax2_3,
            )
            ax2_3.set_title("Task Distance Bins vs Signal Starvation Level", fontsize=13, fontweight="bold")
            ax2_3.set_xlabel("Distance Bin (m)", fontsize=11)
            ax2_3.set_ylabel("Number of Tasks", fontsize=11)
            ax2_3.legend(title="Level", frameon=True)

            plt.tight_layout()
            task_plot_path = out_dir / "task_distribution.png"
            fig2.savefig(task_plot_path, dpi=300)
            plt.close(fig2)
            generated_plots.append(task_plot_path)

        return generated_plots

    def print_summary(
        self,
        df_maps: pd.DataFrame,
        df_tasks: Optional[pd.DataFrame] = None,
    ) -> None:
        """Print concise console summary with classification tables."""
        total_maps = len(df_maps)
        p33 = float(df_maps["tertile_p33"].iloc[0])
        p67 = float(df_maps["tertile_p67"].iloc[0])

        print("\n" + "=" * 78)
        print("[REPORT] NAVSIM MAP TOPOLOGICAL COMPLEXITY (grid-turtlebot4-v2)")
        print("=" * 78)
        print(f"Total Maps Analyzed: {total_maps}")
        print(f"Dataset Root:       {self.dataset_root}")
        print(f"main_m2 range:      [{df_maps['main_m2'].min():.2f}, {df_maps['main_m2'].max():.2f}] m2 (mean: {df_maps['main_m2'].mean():.2f} +/- {df_maps['main_m2'].std():.2f})")
        print(f"obstacle_density:   [{df_maps['obstacle_density'].min():.4f}, {df_maps['obstacle_density'].max():.4f}] (mean: {df_maps['obstacle_density'].mean():.4f})")
        print(f"free_cells:         [{df_maps['free_cells'].min()}, {df_maps['free_cells'].max()}] (mean: {df_maps['free_cells'].mean():.1f})")

        print("\n--- 1. BANG PHAN NHOM MAP THEO KICH THUOC (main_m2) ---")
        print("Phuong phap Tertile (33.3% / 66.7%):")
        t_counts = df_maps["map_complexity"].value_counts().reindex(["Small", "Medium", "Large"])
        print(f"| {'Nhom':<8} | {'Tieu chi':<25} | {'So maps':<10} | {'Ty le':<8} |")
        print(f"|{'-'*10}|{'-'*27}|{'-'*12}|{'-'*10}|")
        print(f"| {'Small':<8} | main_m2 < {p33:.2f} m2            | {t_counts['Small']:<10} | {t_counts['Small']/total_maps*100:.1f}%    |")
        print(f"| {'Medium':<8} | {p33:.2f} <= main_m2 < {p67:.2f} m2     | {t_counts['Medium']:<10} | {t_counts['Medium']/total_maps*100:.1f}%    |")
        print(f"| {'Large':<8} | main_m2 >= {p67:.2f} m2           | {t_counts['Large']:<10} | {t_counts['Large']/total_maps*100:.1f}%    |")

        print("\nPhuong phap Nguong Tron (< 30, 30-50, >= 50 m2):")
        r_counts = df_maps["map_complexity_round"].value_counts().reindex(["Small", "Medium", "Large"])
        print(f"| {'Nhom':<8} | {'Tieu chi':<25} | {'So maps':<10} | {'Ty le':<8} |")
        print(f"|{'-'*10}|{'-'*27}|{'-'*12}|{'-'*10}|")
        print(f"| {'Small':<8} | main_m2 < 30.00 m2            | {r_counts['Small']:<10} | {r_counts['Small']/total_maps*100:.1f}%    |")
        print(f"| {'Medium':<8} | 30.00 <= main_m2 < 50.00 m2    | {r_counts['Medium']:<10} | {r_counts['Medium']/total_maps*100:.1f}%    |")
        print(f"| {'Large':<8} | main_m2 >= 50.00 m2           | {r_counts['Large']:<10} | {r_counts['Large']/total_maps*100:.1f}%    |")

        if df_tasks is not None and not df_tasks.empty:
            total_tasks = len(df_tasks)
            print("\n" + "=" * 78)
            print("[REPORT] 2. BANG PHAN LOAI TASKS THEO MUC DO KHO (SIGNAL STARVATION)")
            print("Chu y: Map khong dai dien cho do kho moi truong. sa_uniform phan anh chinh xac MDP.")
            print("=" * 78)

            grouped = df_tasks.groupby("level")
            print(f"| {'Nhom':<9} | {'Tieu chi sa_uniform':<26} | {'So tasks':<10} | {'Ty le':<8} | {'d_bfs (mean)':<12} | {'sa_uniform (mean)':<18} |")
            print(f"|{'-'*11}|{'-'*28}|{'-'*12}|{'-'*10}|{'-'*14}|{'-'*20}|")
            for lvl, crit in [
                ("easy", "sa_uniform >= 0.10"),
                ("medium", "0.001 <= sa_uniform < 0.10"),
                ("hard", "0.0001 <= sa_uniform < 0.001"),
                ("starved", "sa_uniform < 0.0001"),
            ]:
                if lvl in grouped.groups:
                    g = grouped.get_group(lvl)
                    cnt = len(g)
                    pct = (cnt / total_tasks) * 100
                    d_mean = g["d_bfs"].mean()
                    sa_mean = g["sa_uniform"].mean()
                    print(f"| {lvl:<9} | {crit:<26} | {cnt:<10} | {pct:>5.1f}%   | {d_mean:>10.1f}   | {sa_mean:>18.6f} |")


def main():
    """Run full map and task profiling pipeline."""
    profiler = MapProfiler()
    df_maps = profiler.profile_maps()
    df_tasks = profiler.profile_tasks()

    # Export CSVs
    csv_map = profiler.export_csv(df_maps, "audit_reports/map_complexity_profile.csv")
    print(f"[OK] Saved map complexity profile to: {csv_map}")

    if not df_tasks.empty:
        csv_task = profiler.export_task_csv(df_tasks, "audit_reports/task_complexity_profile.csv")
        print(f"[OK] Saved task complexity profile to: {csv_task}")

    # Plot distributions
    plots = profiler.plot_distributions(df_maps, df_tasks, "audit_reports")
    for p in plots:
        print(f"[OK] Saved plot to: {p}")

    # Print summary tables
    profiler.print_summary(df_maps, df_tasks)


if __name__ == "__main__":
    main()
