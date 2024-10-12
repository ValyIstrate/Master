import gzip
import pickle
import numpy as np
import torch


class DatasetWrapper:
    def __init__(self, train_set_elements, train_set_labels):
        self.elements = train_set_elements
        self.labels = train_set_labels


def get_datasets():
    with gzip.open("mnist.pkl.gz", "rb") as fd:
        initial_train_set, valid_set, initial_test_set = pickle.load(fd, encoding="latin")

    # Combined the train set and the test set
    train_set_x = torch.tensor(np.concatenate([initial_train_set[0], initial_test_set[0]]), dtype=torch.float32)
    train_set_y = torch.tensor(np.concatenate([initial_train_set[1], initial_test_set[1]]), dtype=torch.long)

    valid_set_x = torch.tensor(valid_set[0], dtype=torch.float32)
    valid_set_y = torch.tensor(valid_set[1], dtype=torch.long)

    print(f'Loaded training dataset: {train_set_x.shape}')
    print(f'Loaded validation dataset: {valid_set_x.shape}')

    return DatasetWrapper(train_set_x, train_set_y), DatasetWrapper(valid_set_x, valid_set_y)
