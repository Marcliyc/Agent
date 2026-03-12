from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    data_path: Path = Path('data/data.pq')
    output_dir: Path = Path('Q2/outputs')
    label_columns: tuple[str, ...] = tuple(f'Y{i}' for i in range(1, 13))
    feature_columns: tuple[str, ...] = tuple(f'X{i}' for i in range(1, 301))
    base_columns: tuple[str, ...] = (
        'trade_date', 'underlying', 'start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume'
    )
    random_seed: int = 42
    top_k: int = 50
    corr_threshold: float = 0.90

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(parents=True, exist_ok=True)
