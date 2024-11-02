import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from torch import GradScaler


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.epochs = config['training']['epochs']
        self.early_stopping_patience = config['training']['early_stopping']

        optim_name = config['training']['optimizer']
        self.optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=config['training']['learning_rate'])

        scheduler_name = config['training']['scheduler']
        if scheduler_name == "StepLR":
            self.scheduler = StepLR(self.optimizer, **config['training']['scheduler_params'])
        elif scheduler_name == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer)
        else:
            self.scheduler = None

        self.tb_writer = SummaryWriter(log_dir=config['logging'].get('tensorboard_log_dir', 'runs'))

        wandb.init(
            project=config['logging'].get('wandb_project_name', 'default_project'),
            config=config
        )
        wandb.watch(self.model, log='all')

        self.pin_memory = True
        self.enable_half = device == 'cuda'
        self.scaler = GradScaler(device, enabled=self.enable_half)


    def train(self):
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        correct = 0
        total = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.enable_half):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return 100.0 * correct / total

    @torch.inference_mode()
    def val(self):
        self.model.eval()
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.enable_half):
                outputs = self.model(inputs)

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    @torch.inference_mode()
    def inference(self):
        self.model.eval()

        labels = []

        for inputs, _ in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.enable_half):
                outputs = self.model(inputs)

            predicted = outputs.argmax(1).tolist()
            labels.extend(predicted)

        return labels

    def train_loop(self):
        best = 0.0
        epochs = list(range(self.epochs))
        with tqdm(epochs) as tbar:
            for epoch in tbar:
                train_acc = self.train()
                val_acc = self.val()
                if val_acc > best:
                    best = val_acc
                tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}")
