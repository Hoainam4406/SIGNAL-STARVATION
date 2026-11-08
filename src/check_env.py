import os
import sys
from pathlib import Path

# Tự động nạp file .env từ thư mục gốc dự án (C:\SIGNAL-STARVATION\.env)
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
    else:
        load_dotenv()
except ImportError:
    pass

def check_environment():
    print("=" * 60)
    print("ENVIRONMENT & DEPENDENCY VERIFICATION")
    print("=" * 60)

    # 1. Environment Variable Check
    navsim_datasets = os.environ.get("NAVSIM_DATASETS")
    expected_path = r"F:\navsim\datasets"
    print(f"\n[Environment Variables]")
    if navsim_datasets:
        print(f"NAVSIM_DATASETS: {navsim_datasets}")
        if os.path.exists(navsim_datasets):
            print(f"  -> Path exists: True ({navsim_datasets})")
        else:
            print(f"  -> Warning: Path does not exist on disk: {navsim_datasets}")
    else:
        print(f"NAVSIM_DATASETS: None")
        print(f"  -> Warning: NAVSIM_DATASETS is not set in current process environment.")
        print(f"     Setting temporary fallback to {expected_path}")
        os.environ["NAVSIM_DATASETS"] = expected_path

    # 2. Package Imports and Version Check
    packages = [
        ("navsim", "navsim"),
        ("gymnasium", "gymnasium"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("numba", "numba"),
        ("matplotlib", "matplotlib"),
        ("pettingzoo", "pettingzoo"),
        ("torch", "torch"),
        ("stable_baselines3", "stable-baselines3"),
        ("sb3_contrib", "sb3-contrib"),
        ("pandas", "pandas"),
        ("seaborn", "seaborn"),
        ("tensorboard", "tensorboard"),
        ("yaml", "pyyaml"),
        ("omegaconf", "omegaconf"),
        ("statsmodels", "statsmodels"),
    ]

    print(f"\n[Packages & Dependencies]")
    all_passed = True
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {display_name:<20} : {version}")
        except Exception as e:
            print(f"  [FAILED] {display_name:<20} : {e}")
            all_passed = False

    # 3. PyTorch CUDA / Device Check
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_avail else 0
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
        print(f"\n[PyTorch Acceleration]")
        print(f"  CUDA Available : {cuda_avail}")
        print(f"  Device Count   : {device_count}")
        print(f"  Active Device  : {device_name}")
    except Exception as e:
        print(f"  Could not inspect torch device: {e}")

    print("=" * 60)
    if all_passed:
        print("RESULT: ALL REQUIRED PACKAGES VERIFIED SUCCESSFULLY!")
    else:
        print("RESULT: SOME PACKAGES FAILED VERIFICATION.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    check_environment()
