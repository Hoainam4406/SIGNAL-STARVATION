import json
from pathlib import Path
import gymnasium as gym
import navsim

datasets_dir = Path(r"F:\navsim\datasets")
ds_folders = sorted([p for p in datasets_dir.iterdir() if p.is_dir() and p.name.startswith("grid-")])

print(f"Total datasets found: {len(ds_folders)}\n")

dataset_stats = []
robots_info = {}

for ds in ds_folders:
    manifest_path = ds / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    splits = manifest.get("splits", {})
    train_maps = splits.get("train", [])
    val_maps = splits.get("val", [])
    test_maps = splits.get("test", [])
    
    task_files = list((ds / "tasks").glob("*.json"))
    grid_files = list((ds / "grids").glob("*.json"))
    
    # Check total tasks count across all task files or inside task files
    # Let's count tasks in task files
    total_tasks = 0
    tasks_by_split = {"train": 0, "val": 0, "test": 0}
    for tf in task_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                tdata = json.load(f)
                # tdata is list or dict with tasks
                if isinstance(tdata, list):
                    n_t = len(tdata)
                elif isinstance(tdata, dict) and "tasks" in tdata:
                    n_t = len(tdata["tasks"])
                else:
                    n_t = 1
                total_tasks += n_t
                # check which split tf belongs to
                map_id = tf.stem
                if map_id in train_maps:
                    tasks_by_split["train"] += n_t
                elif map_id in val_maps:
                    tasks_by_split["val"] += n_t
                elif map_id in test_maps:
                    tasks_by_split["test"] += n_t
        except Exception:
            pass

    dataset_stats.append({
        "name": ds.name,
        "train_maps": len(train_maps),
        "val_maps": len(val_maps),
        "test_maps": len(test_maps),
        "total_maps": len(train_maps) + len(val_maps) + len(test_maps),
        "grid_files": len(grid_files),
        "task_files": len(task_files),
        "tasks_by_split": tasks_by_split,
        "total_tasks": total_tasks
    })
    
    # Read robot JSON
    for rf in (ds / "robots").glob("*.json"):
        if rf.name not in robots_info:
            with open(rf, "r", encoding="utf-8") as f:
                robots_info[rf.name] = json.load(f)

print("=== 1. DATASET SPLIT STATS ===")
for s in dataset_stats:
    print(f"{s['name']}:")
    print(f"  Maps: train={s['train_maps']}, val={s['val_maps']}, test={s['test_maps']} (Total: {s['total_maps']})")
    print(f"  Tasks: train={s['tasks_by_split']['train']}, val={s['tasks_by_split']['val']}, test={s['tasks_by_split']['test']} (Total: {s['total_tasks']})")

print("\n=== 2. ROBOTS SPECIFICATIONS (9 ROBOTS) ===")
for rfile, rdata in sorted(robots_info.items()):
    rid = rdata.get("id")
    rname = rdata.get("name")
    fp = rdata.get("footprint_simple", {})
    if not fp:
        fp = rdata.get("footprint", {})
    limits = rdata.get("limits", {})
    lidar = rdata.get("lidar", {})
    height = rdata.get("height")
    print(f"- ID: {rid}")
    print(f"  Name: {rname}")
    print(f"  Footprint Simple: {fp}")
    print(f"  Height: {height} m (class: {rdata.get('height_class')})")
    print(f"  Drive: {rdata.get('drive')}")
    print(f"  Limits: v_max={limits.get('v_max')} m/s, v_min={limits.get('v_min')} m/s, w_max={limits.get('w_max')} rad/s")
    print(f"  LiDAR: range=[{lidar.get('range_min')}, {lidar.get('range_max')}] m, fov={lidar.get('fov')} rad, beams={lidar.get('beams')}")

print("\n=== 3. GRID METRICS SAMPLE ===")
# Inspect sample grid files across datasets
sample_grid = ds_folders[0] / "grids"
sample_files = list(sample_grid.glob("*.json"))[:3]
for sf in sample_files:
    with open(sf, "r", encoding="utf-8") as f:
        g = json.load(f)
    print(f"Grid {sf.stem}: W={g.get('W')}, H={g.get('H')}, cell_m={g.get('cell_m')}, free_cells={g.get('free_cells')}, main_cells={g.get('main_cells')}, main_frac={g.get('main_frac')}, main_m2={g.get('main_m2')}")

print("\n=== 4. TEST NavSim/GridNav-v0 SPACES & INFO ===")
try:
    env_grid = gym.make("NavSim/GridNav-v0")
    print(f"GridNav-v0 Action Space: {env_grid.action_space}")
    print(f"GridNav-v0 Obs Space: {env_grid.observation_space}")
    obs, info = env_grid.reset()
    print(f"GridNav-v0 Reset Info Keys: {list(info.keys())}")
    obs, reward, done, trunc, info = env_grid.step(env_grid.action_space.sample())
    print(f"GridNav-v0 Step Info Keys: {list(info.keys())}")
    env_grid.close()
except Exception as e:
    print(f"Error testing GridNav-v0: {e}")

print("\n=== 5. TEST NavSim/Nav-v0 (SVG continuous) WITH demo-svg ===")
try:
    env_svg = gym.make("NavSim/Nav-v0", dataset="demo-svg")
    print(f"Nav-v0 Action Space: {env_svg.action_space}")
    print(f"Nav-v0 Obs Space: {env_svg.observation_space}")
    obs_svg, info_svg = env_svg.reset()
    print(f"Nav-v0 Reset Info Keys: {list(info_svg.keys())}")
    obs_svg, reward_svg, done_svg, trunc_svg, info_svg = env_svg.step(env_svg.action_space.sample())
    print(f"Nav-v0 Step Successful! Reward: {reward_svg}, Done: {done_svg}")
    env_svg.close()
    print("Nav-v0 status: FUNCTIONAL WITH demo-svg")
except Exception as e:
    print(f"Error testing Nav-v0: {e}")
