import os
import shutil
import tomllib
from pathlib import Path


def _find_config_path(config_name: str) -> Path:
    return (
        Path(os.environ.get("CONFIG_DIR", os.path.expanduser("~/.config")))
        / "causal_recovery"
        / config_name
    )


def _find_package_template_path() -> Path:
    current_file = Path(__file__)
    package_config_dir = current_file.parent
    return package_config_dir / "config.toml.template"


class Config:
    def __init__(
        self,
        config_name: str | None = None,
        template_path: Path | None = None,
    ):
        if config_name is None:
            config_name = os.environ.get("CAUSAL_RECOVERY_CONFIG", "config.toml")

        self.config_path = _find_config_path(config_name)

        if template_path is not None:
            self.template_path = Path(template_path)
        else:
            self.template_path = _find_package_template_path()

        if not self.config_path.exists():
            if self.template_path.exists():
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(self.template_path, self.config_path)
                os.chmod(self.config_path, 0o600)
                print(
                    f"Created {self.config_path} from template at {self.template_path}. "
                    "Please review and adjust settings."
                )
            else:
                raise FileNotFoundError(
                    f"Neither {self.config_path} nor {self.template_path} found. "
                    "Please create a configuration file or specify a valid template path."
                )

        self._load_config()

    def _load_config(self):
        with open(self.config_path, "rb") as f:
            self.config = tomllib.load(f)


config = Config()
