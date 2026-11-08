import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union
import yaml
from omegaconf import OmegaConf, DictConfig


class ConfigManager:
    """Quản lý cấu hình thực nghiệm từ tệp YAML, hỗ trợ hash và lưu snapshot."""

    def __init__(self, config_data: Union[str, Path, Dict[str, Any], DictConfig] = None):
        self.config_dict: Dict[str, Any] = {}
        self.cfg: DictConfig = OmegaConf.create()
        self.source_path: Union[Path, None] = None

        if config_data is not None:
            self.load(config_data)

    def load(self, source: Union[str, Path, Dict[str, Any], DictConfig]) -> "ConfigManager":
        """Nạp cấu hình từ đường dẫn tệp YAML hoặc dictionary."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {path}")
            self.source_path = path
            loaded = OmegaConf.load(str(path))
            self.cfg = loaded
            self.config_dict = OmegaConf.to_container(loaded, resolve=True)
        elif isinstance(source, dict):
            self.cfg = OmegaConf.create(source)
            self.config_dict = source
        elif isinstance(source, DictConfig):
            self.cfg = source
            self.config_dict = OmegaConf.to_container(source, resolve=True)
        else:
            raise ValueError(f"Unsupported config source type: {type(source)}")
        return self

    def get_hash(self) -> str:
        """Sinh mã băm SHA-256 từ nội dung cấu hình để đảm bảo audit trail."""
        serialized = json.dumps(self.config_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def save_snapshot(self, output_dir: Union[str, Path], filename: str = "config_snapshot.yaml") -> Path:
        """Lưu bản snapshot của cấu hình vào thư mục kết quả thực nghiệm."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target_path = out_dir / filename
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return target_path

    def get(self, key: str, default: Any = None) -> Any:
        """Lấy giá trị theo key dạng 'agent.learning_rate'."""
        keys = key.split(".")
        val = self.config_dict
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def to_dict(self) -> Dict[str, Any]:
        return self.config_dict
