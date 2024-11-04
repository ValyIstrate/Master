from .cifar import ResNet18_CIFAR10
from .mnist import MLP, LeNet


def load_model(config):
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']

    if model_name == "resnet18_cifar10":
        return ResNet18_CIFAR10(num_classes=num_classes)
    elif model_name == "preact_resnet18":
        return ResNet18_CIFAR10(num_classes=num_classes)
    elif model_name == "mlp":
        return MLP(num_classes=num_classes)
    elif model_name == "lenet":
        return LeNet(num_classes=num_classes)
    else:
        raise Exception(f"Model {model_name} is not supported.")
