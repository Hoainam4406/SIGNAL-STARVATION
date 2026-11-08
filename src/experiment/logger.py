import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Union

from torch.utils.tensorboard import SummaryWriter

from .config_manager import ConfigManager


class ExperimentLogger:
    """
    Hệ thống ghi nhận metrics huấn luyện và đánh giá thực nghiệm (Audit Trail).
    - Quản lý cấu trúc thư mục results/<experiment_name>/<timestamp>/
    - Tích hợp TensorBoard logging thời gian thực
    - Ghi train_metrics.csv và chuẩn file eval_results.csv
    - Lưu vết metadata: git commit hash, config hash, seed
    """

    EVAL_CSV_COLUMNS: List[str] = [
        "experiment_id",
        "seed",
        "algorithm",
        "dataset",
        "robot",
        "reward_type",
        "level_filter",
        "episode",
        "total_reward",
        "success",
        "steps_to_goal",
        "collision_count",
        "spl",
        "mean_progress",
        "sa_uniform_mean",
        "training_time_s",
        "eval_timestamp",
    ]

    TRAIN_CSV_COLUMNS: List[str] = [
        "episode",
        "reward",
        "steps",
        "time",
    ]

    def __init__(
        self,
        experiment_name: str = "default_experiment",
        seed: int = 42,
        algorithm: str = "PPO",
        dataset: str = "navsim_mini",
        robot: str = "differential_drive",
        reward_type: str = "sparse",
        level_filter: str = "all",
        config: Optional[Union[Dict[str, Any], ConfigManager, str, Path]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        use_tensorboard: bool = True,
    ):
        self.experiment_name = experiment_name
        self.seed = seed
        self.algorithm = algorithm
        self.dataset = dataset
        self.robot = robot
        self.reward_type = reward_type
        self.level_filter = level_filter
        self.use_tensorboard = use_tensorboard
        self.start_time = time.time()

        # Xác định base_dir: mặc định là <project_root>/results
        if base_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.base_dir = project_root / "results"
        else:
            self.base_dir = Path(base_dir)

        # Định dạng timestamp và experiment_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{self.experiment_name}_{self.algorithm}_s{self.seed}_{self.timestamp}"

        # Tạo thư mục: results/<experiment_name>/<timestamp>/ (tự động phân biệt nếu chạy cùng giây)
        self.run_dir = self.base_dir / self.experiment_name / self.timestamp
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            self.run_dir = self.base_dir / self.experiment_name / f"{self.timestamp}_s{self.seed}"

        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Xử lý config và config snapshot
        self.config_manager: Optional[ConfigManager] = None
        self.config_hash: str = "none"
        if config is not None:
            if isinstance(config, ConfigManager):
                self.config_manager = config
            else:
                self.config_manager = ConfigManager(config)
            self.config_hash = self.config_manager.get_hash()
            self.config_manager.save_snapshot(self.run_dir, "config_snapshot.yaml")

        # Thu thập Git Commit Hash
        self.git_commit_hash = self._get_git_commit_hash()

        # Lưu metadata thực nghiệm
        self._save_metadata()

        # Khởi tạo TensorBoard
        self.tb_writer: Optional[SummaryWriter] = None
        if self.use_tensorboard:
            tb_log_dir = self.run_dir / "tensorboard"
            self.tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

        # Khởi tạo các file CSV
        self.train_csv_path = self.run_dir / "train_metrics.csv"
        self._init_csv(self.train_csv_path, self.TRAIN_CSV_COLUMNS)

        self.eval_csv_path = self.run_dir / "eval_results.csv"
        self._init_csv(self.eval_csv_path, self.EVAL_CSV_COLUMNS)

    def _get_git_commit_hash(self) -> str:
        """Lấy Git commit hash hiện tại của repository."""
        try:
            repo_dir = Path(__file__).resolve().parent.parent.parent
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except Exception:
            return "unknown_or_no_git"

    def _save_metadata(self) -> None:
        """Ghi metadata.json đầy đủ phục vụ audit trail."""
        metadata = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "robot": self.robot,
            "reward_type": self.reward_type,
            "level_filter": self.level_filter,
            "config_hash": self.config_hash,
            "git_commit_hash": self.git_commit_hash,
            "run_dir": str(self.run_dir),
        }
        with open(self.run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def _init_csv(self, file_path: Path, headers: List[str]) -> None:
        """Khởi tạo file CSV với tiêu đề cột nếu chưa tồn tại."""
        if not file_path.exists():
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_train_metric(self, episode: int, reward: float, steps: int, elapsed_time: Optional[float] = None) -> None:
        """
        Ghi lại metric sau mỗi episode huấn luyện:
        - train_metrics.csv: episode, reward, steps, time
        - TensorBoard scalars
        """
        if elapsed_time is None:
            elapsed_time = time.time() - self.start_time

        with open(self.train_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, steps, round(elapsed_time, 4)])

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/episode_reward", reward, episode)
            self.tb_writer.add_scalar("train/episode_steps", steps, episode)
            self.tb_writer.add_scalar("train/elapsed_time_s", elapsed_time, episode)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Ghi một giá trị scalar tùy chọn lên TensorBoard."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)

    def log_eval_result(
        self,
        episode: int,
        total_reward: float,
        success: Union[bool, int, float],
        steps_to_goal: int,
        collision_count: int = 0,
        spl: float = 0.0,
        mean_progress: float = 0.0,
        sa_uniform_mean: float = 0.0,
        training_time_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Ghi dòng kết quả đánh giá theo đúng chuẩn:
        experiment_id, seed, algorithm, dataset, robot, reward_type, level_filter,
        episode, total_reward, success, steps_to_goal, collision_count,
        spl, mean_progress, sa_uniform_mean, training_time_s, eval_timestamp
        """
        if training_time_s is None:
            training_time_s = time.time() - self.start_time

        eval_timestamp = datetime.now().isoformat()
        row = {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "robot": self.robot,
            "reward_type": self.reward_type,
            "level_filter": self.level_filter,
            "episode": episode,
            "total_reward": round(float(total_reward), 4),
            "success": int(bool(success)),
            "steps_to_goal": steps_to_goal,
            "collision_count": collision_count,
            "spl": round(float(spl), 4),
            "mean_progress": round(float(mean_progress), 4),
            "sa_uniform_mean": round(float(sa_uniform_mean), 4),
            "training_time_s": round(float(training_time_s), 4),
            "eval_timestamp": eval_timestamp,
        }

        # Ghi vào CSV
        with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.EVAL_CSV_COLUMNS)
            writer.writerow(row)

        # Ghi các metric đánh giá chính lên TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("eval/total_reward", total_reward, episode)
            self.tb_writer.add_scalar("eval/success_rate", float(row["success"]), episode)
            self.tb_writer.add_scalar("eval/spl", float(spl), episode)
            self.tb_writer.add_scalar("eval/collision_count", collision_count, episode)
            self.tb_writer.add_scalar("eval/mean_progress", float(mean_progress), episode)

        return row

    def close(self) -> None:
        """Đóng các luồng ghi log, đảm bảo toàn bộ dữ liệu đã được flush xuống đĩa."""
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
