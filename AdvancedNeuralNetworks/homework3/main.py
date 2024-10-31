import argparse
import yaml
import torch
from trainer.trainer import Trainer
from data.dataset import get_dataloader
from model.model_loader import load_model
from utils.log import simple_text, info, warn
from torcheval.metrics.functional import multiclass_accuracy
from torcheval.metrics.functional import multiclass_f1_score


def write_data_set_scores(prediction: torch.Tensor, labels):
    prediction = torch.tensor(prediction) if not isinstance(prediction, torch.Tensor) else prediction
    labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels

    accuracy = multiclass_accuracy(prediction, labels)
    f1_score = multiclass_f1_score(prediction, labels)

    simple_text(f"Accuracy: {accuracy}")
    simple_text(f"F1-Score: {f1_score}")

    simple_text("All Classes Accuracy:")
    class_accuracies = multiclass_accuracy(prediction, labels, average=None, num_classes=10)
    for cls, value in enumerate(class_accuracies):
        simple_text(f"    Class {cls}: {value}")

    simple_text("All Classes F1-Score:")
    class_f1_scores = multiclass_f1_score(prediction, labels, average=None, num_classes=10)
    for cls, value in enumerate(class_f1_scores):
        simple_text(f"    Class {cls}: {value}")


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

    # trainer.train_loop()
    trainer.train()

    # predictions, labels = trainer.get_final_predictions(val_loader)
    # write_data_set_scores(predictions, labels)

if __name__ == "__main__":
    main()
