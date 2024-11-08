import wandb
from trainer.trainer import Trainer
from model.model_loader import load_model
from data.dataset import get_dataloader
import torch
from utils.log import info, warn
import yaml


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


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sweep_config():
    return {
        'method': 'grid',
        'metric': {
            'name': 'validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'scheduler': {
                'values': ['ReduceLROnPlateau', "StepLR"]
            },
            'learning_rate': {
                'values': [0.001, 0.1]
            },
            'optimizer': {
                'values': ['ADAM', 'SGD']
            }
        }
    }
def main():
    config = load_config('config/config.yaml')
    device = torch.device(load_device(config))

    run = wandb.init()
    trainer = Trainer(model, train_loader, val_loader, config, device, run.config)
    trainer.train_loop()


wandb.login()
sweep_config = load_sweep_config()

config = load_config('/kaggle/input/config/config.yaml')
#config = load_config('config.yaml')
device = torch.device(load_device(config))

model = load_model(config).to(device)
train_loader, val_loader = get_dataloader(config)

# wandb.init(project='tema3', config=sweep_config)

sweep_id = wandb.sweep(sweep=sweep_config, project="tema3")

wandb.agent(sweep_id, function=main, count=8)

wandb.finish()