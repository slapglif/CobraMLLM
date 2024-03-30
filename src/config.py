import json

from pydantic import BaseModel


class Config(BaseModel):
    vision_size: int = 2048
    text_size: int = 10000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    learning_rate: float = 0.0001
    batch_size: int = 32
    num_epochs: int = 10
    dataset_type: str = "hf"
    dataset_name: str = "gondimjoaom/flickr30k-trainready"
    train_split: str = "train"
    val_split: str = "validation"
    local_train_data_path: str = "path/to/local/train/data"
    local_val_data_path: str = "path/to/local/val/data"
    num_workers: int = 4
    weight_decay: float = 0.01
    patience: int = 3
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 5
    shuffle: bool = True

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
