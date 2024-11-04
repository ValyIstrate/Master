import torch


def save(model, file_path: str, optimizer=None, epoch=None):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
    }
    torch.save(state, file_path)
