import torch


def init_kaiming_uniform(tensor, fan_in):
    """
    Because we are using the ReLU activation function, initializing the weights using the
    Kaiming uniform method is recommended.
    https://ai.stackexchange.com/questions/32247/what-is-the-analytical-formula-for-kaiming-he-probability-density-function
    :param tensor: The tensor we want to initialize the weights with.
    :param fan_in: Number of input units
    :return: The initialized weights
    """
    gain = torch.sqrt(torch.tensor(2.0))
    std = gain / torch.sqrt(torch.tensor(fan_in, dtype=torch.float32))
    bound = torch.sqrt(torch.tensor(3.0)) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound)


def relu(x):
    return torch.maximum(x, torch.tensor(0.0))


def softmax(x, dim=1):
    exps = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exps / exps.sum(dim=dim, keepdim=True)


class DigitClassifier:
    def __init__(self):
        # These values will be used later
        self.output = None
        self.hidden_activation = None
        self.hidden_output = None

        # Input layer - 784 units
        # Hidden Layer - 100 units
        self.hidden_weight = torch.zeros(784, 100)
        self.hidden_bias = torch.zeros(100)

        # Hidden layer - 100 units
        # Output layer - 10 units
        self.output_weight = torch.zeros(100, 10)
        self.output_bias = torch.zeros(10)

        # Initialize the weights
        init_kaiming_uniform(self.hidden_weight, 784)
        init_kaiming_uniform(self.output_weight, 100)

    def forward(self, x):
        # Manual forward propagation
        self.hidden_output = x @ self.hidden_weight + self.hidden_bias
        # Use ReLU for the hidden layer
        self.hidden_activation = relu(self.hidden_output)
        self.output = self.hidden_activation @ self.output_weight + self.output_bias
        # Use softmax for the output layer
        return softmax(self.output, dim=1)


    def backward(self, x, y, output):
        batch_size = y.size(0)

        # Backpropagation through the output layer (softmax + gradient descent)
        d_output = output.clone()
        d_output[range(batch_size), y] -= 1  # Gradient of softmax and cross-entropy combined
        d_output /= batch_size

        # Gradients calculated based on output layer weights and biases
        dw_output = self.hidden_activation.T @ d_output
        db_output = d_output.sum(dim=0)

        # Backpropagation through the hidden layer
        d_hidden_activation = d_output @ self.output_weight.T
        d_hidden_output = d_hidden_activation.clone()
        # ReLU derivative: 0 if input <= 0, 1 otherwise
        d_hidden_output[self.hidden_output <= 0] = 0

        dw_hidden = x.T @ d_hidden_output
        db_hidden = d_hidden_output.sum(dim=0)

        # Manually set gradients
        self.hidden_weight.grad = dw_hidden
        self.hidden_bias.grad = db_hidden
        self.output_weight.grad = dw_output
        self.output_bias.grad = db_output
