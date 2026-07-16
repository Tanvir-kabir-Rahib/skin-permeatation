from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def src(self) -> Path:
        return self.root / "src"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw_data(self) -> Path:
        return self.data / "raw"

    @property
    def processed_data(self) -> Path:
        return self.data / "processed"

    @property
    def final_data(self) -> Path:
        return self.data / "final"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def descriptors_generator(self) -> Path:
        return self.root / "descriptors-generator"

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"

    @property
    def tests(self) -> Path:
        return self.root / "tests"

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProjectPaths":
        start_path = (start or Path.cwd()).resolve()
        for candidate in [start_path, *start_path.parents]:
            if (candidate / "data").exists() and (candidate / "src").exists():
                return cls(candidate)
        raise FileNotFoundError("Could not locate project root containing both 'data' and 'src'.")

    def ensure_runtime_dirs(self) -> None:
        for directory in [
            self.reports,
            self.figures,
            self.models / "reproduction",
            self.reports / "tables",
            self.reports / "logs",
            self.reports / "artifacts",
        ]:
            directory.mkdir(parents=True, exist_ok=True)
