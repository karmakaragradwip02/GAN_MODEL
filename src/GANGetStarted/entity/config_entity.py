from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen= True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen= True)
class DataPreparationConfig:
    root_dir: Path
    data_dir: Path
    monet_dir: Path
    photo_dir: Path

@dataclass(frozen= True)
class ModelPreparationConfig:
    root_dir: Path
    model_dir: Path
    weight_decay: float
    input_image_size: list
    learning_rate: float
    epsilon: float
    decay_rate: float

@dataclass(frozen= True)
class ModelTrainingConfig:
    root_dir: Path
    gan_model_dir: Path
    disc_model_dir: Path
    trained_model_dir: Path
    history_dir: Path
    epochs: int

@dataclass(frozen= True)
class ModelEvaluationConfig:
    root_dir: Path
    trained_model_dir: Path
    history_dir: Path
    graph_dir: Path
    mlflow_uri: str
    all_params: dict
    epochs: int