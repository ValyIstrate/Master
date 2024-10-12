from datetime import datetime

from torcheval.metrics.functional import multiclass_accuracy
from torcheval.metrics.functional import multiclass_f1_score

import data_handler
from data_handler import DatasetWrapper
from DigitClassifier import *


def write_data_set_scores(prediction, file, dataset_wrapper: DatasetWrapper):
    """
    Method writes the training results in a file.
    :param prediction: The predictions on the dataset, at the end of the training loop.
    :param file: The file in which the results should be written to.
    :param dataset_wrapper: The class containing the dataset elements and labels
    :return: Void method
    """

    file.write(f"Accuracy: {multiclass_accuracy(prediction, dataset_wrapper.labels)}\n")
    file.write(f"F1-Score: {multiclass_f1_score(prediction, dataset_wrapper.labels)}\n")
    file.write(f"All Classes Accuracy: \n")
    for cls, value in enumerate(multiclass_accuracy(prediction, dataset_wrapper.labels, average=None, num_classes=10)):
        file.write(f"     {cls}: {value}\n")
    file.write(f"All Classes F1-Score: \n")
    for cls, value in enumerate(multiclass_f1_score(prediction, dataset_wrapper.labels, average=None, num_classes=10)):
        file.write(f"     {cls}: {value}\n")


def cross_entropy_loss(output, target):
    """
    Method computes the cross entropy loss.
    :param output: This is the predicted output from the model, typically after applying softmax
    :param target: This is a tensor containing the true class labels (integers) for each sample, of shape.
    :return: Returns the cross entropy loss.
    """
    # log_probs = torch.log(output) # This takes the natural logarithm of the predicted probabilities
    # return -torch.mean(torch.gather(log_probs, 1, target.unsqueeze(1)))
    batch_size = output.size(0)
    num_classes = output.size(1)

    log_probs = torch.log(output)  #Calculate the logarithm of the predicted probabilities

    # Create one-hot encoded target tensor of shape (batch_size, num_classes)
    one_hot_target = torch.zeros(batch_size, num_classes).to(output.device)
    one_hot_target[range(batch_size), target] = 1

    # Calculate the cross-entropy loss using the formula in Course 1
    loss = -torch.sum(one_hot_target * log_probs) / batch_size

    return loss


def train_network():
    start = datetime.now()
    train_set, validation_set = data_handler.get_datasets()

    lr = 0.005
    epochs = 100
    batch_size = 32

    model = DigitClassifier()

    for epoch in range(epochs):
        for i in range(0, len(train_set.elements), batch_size):
            images = train_set.elements[i: i + batch_size].view(-1, 784)
            labels = train_set.labels[i: i + batch_size]
            output = model.forward(images)

            model.backward(images, labels, output)

            # Disable autograd for manual update
            with torch.no_grad():
                for param in [model.hidden_weight, model.hidden_bias, model.output_weight, model.output_bias]:
                    param -= lr * param.grad

            # Zero gradients for the next step
            for param in [model.hidden_weight, model.hidden_bias, model.output_weight, model.output_bias]:
                param.grad.zero_()

        print(f'Epoch {epoch + 1} with loss'
              f' {cross_entropy_loss(model.forward(train_set.elements.view(-1, 784)), train_set.labels)}')

    train_set_file = open("train_set.txt", "w")
    valid_set_file = open("valid_set.txt", "w")
    with torch.no_grad():
        predicted_train = model.forward(train_set.elements)
    write_data_set_scores(predicted_train, train_set_file, train_set)
    train_set_file.close()
    with torch.no_grad():
        predicted_valid = model.forward(validation_set.elements)
    write_data_set_scores(predicted_valid, valid_set_file, validation_set)
    valid_set_file.close()

    end = datetime.now()

    print(f'Runtime: {end - start}')


if __name__ == '__main__':
    train_network()