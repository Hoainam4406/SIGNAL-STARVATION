"""Tests for MapProfiler and topological complexity extraction."""

from pathlib import Path
import pytest
import pandas as pd

from src.experiment.map_profiler import MapProfiler


def test_map_profiler_init():
    profiler = MapProfiler(dataset_name="grid-turtlebot4-v2")
    assert profiler.dataset_root.exists()
    assert profiler.grids_dir.exists()


def test_map_profiler_profile_maps():
    profiler = MapProfiler(dataset_name="grid-turtlebot4-v2")
    df_maps = profiler.profile_maps()

    assert isinstance(df_maps, pd.DataFrame)
    assert len(df_maps) == 138

    expected_cols = [
        "map_id", "scene", "storey", "split", "W", "H", "cell_m",
        "free_cells", "main_cells", "main_frac", "obstacle_density",
        "main_m2", "map_complexity", "map_complexity_round",
    ]
    for col in expected_cols:
        assert col in df_maps.columns

    # Verify obstacle density = 1 - main_frac
    diff = (df_maps["obstacle_density"] - (1.0 - df_maps["main_frac"])).abs().max()
    assert diff < 1e-4

    # Verify tertile classification balance
    counts = df_maps["map_complexity"].value_counts()
    assert set(counts.keys()) == {"Small", "Medium", "Large"}
    assert counts["Small"] == 46
    assert counts["Medium"] == 46
    assert counts["Large"] == 46


def test_map_profiler_profile_tasks():
    profiler = MapProfiler(dataset_name="grid-turtlebot4-v2")
    df_tasks = profiler.profile_tasks()

    assert isinstance(df_tasks, pd.DataFrame)
    assert len(df_tasks) == 2760  # 138 maps * 20 tasks each

    expected_task_cols = [
        "map_id", "task_id", "d_bfs", "sa_uniform", "sa_persist", "starved", "level", "dist_bin"
    ]
    for col in expected_task_cols:
        assert col in df_tasks.columns

    # Check task levels match the 4 expected classes
    levels = set(df_tasks["level"].unique())
    assert levels == {"easy", "medium", "hard", "starved"}
