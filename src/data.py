import json
from typing import Dict, Tuple, Any

import torch
from PIL import Image
from datasets import load_dataset
from mpmath.identification import transforms
from torch.utils.data import Dataset, DataLoader


class MultimodalDataset(Dataset):
    """
    A custom dataset class for multimodal data.

    Args:
        data (Dict[str, Any]): The multimodal dataset containing image paths, text, and labels.
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, int]:
        item = self.data[index]
        vision_input = self.load_image(item["image_path"])
        text_input = item["text"]
        label = item["label"]
        return vision_input, text_input, label

    @classmethod
    def load_image(cls, image_path: str) -> torch.Tensor:
        """
        Loads an image from the given path and applies preprocessing.

        Args:
            image_path (str): The path to the image file.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        image = Image.open(image_path).convert("RGB")

        # Define image preprocessing and transformations
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to a fixed size
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])

        # Apply preprocessing and transformations to the image
        image_tensor = preprocess(image)

        return image_tensor


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Creates data loaders for training and validation based on the provided configuration.

    Args:
        config (Dict[str, Any]): The configuration dictionary containing dataset information.

    Returns:
        Tuple[DataLoader, DataLoader]: The train and validation data loaders.
    """
    if config["dataset_type"] == "hf":
        dataset = load_dataset(config["dataset_name"])
        train_dataset = dataset[config["train_split"]]
        val_dataset = dataset[config["val_split"]]
    else:
        train_data = load_local_data(config["local_train_data_path"])
        val_data = load_local_data(config["local_val_data_path"])
        train_dataset = MultimodalDataset(train_data)
        val_dataset = MultimodalDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])

    return train_loader, val_loader


def load_local_data(data_path: str) -> Dict[str, Any]:
    """
    Loads local dataset from the given path.

    Args:
        data_path (str): The path to the local dataset file.

    Returns:
        Dict[str, Any]: The loaded dataset dictionary.
    """
    with open(data_path, "r") as file:
        data = json.load(file)
    return data
