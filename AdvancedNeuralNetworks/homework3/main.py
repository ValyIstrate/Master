import argparse
import yaml
import torch
from trainer.trainer import Trainer
from data.dataset import get_dataloader
from model.model_loader import load_model
from utils.log import info, warn


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training Pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file')
    parser.add_argument('--log-in-file', type=bool, default=False, help='Write logs in log file')
    return parser.parse_args()


def load_device(config) -> str:
    device = config['training']['device']
    if device == 'cuda' or device == 'gpu':
        if torch.cuda.is_available():
            info('Device set: GPU')
            return 'cuda'
        else:
            warn('GPU not available. Device set: CPU')
            return 'cpu'
    info('Device set: CPU')
    return 'cpu'


def main():
    args = parse_args()

    config = load_config(args.config)
    device = torch.device(load_device(config))

    model = load_model(config).to(device)
    train_loader, val_loader = get_dataloader(config)
    trainer = Trainer(model, train_loader, val_loader, config, device)

    trainer.train_loop()

if __name__ == "__main__":
    main()
