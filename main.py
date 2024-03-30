import argparse
from src.config import Config
from src.model import CobraMLLM
from src.train import train
from src.data import create_dataloaders


def main(config_file):
    # Load the configuration from the JSON file
    config = Config.from_json_file(config_file)

    # Load the model from the specified file
    model = CobraMLLM(**config.dict())
    val_loader, train_loader = create_dataloaders(config.dict())
    # Train the model
    train(model, train_loader, val_loader, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-modal language model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration JSON file")
    args = parser.parse_args()

    main(args.config)
