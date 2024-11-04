# Training Pipeline

## How to run

```bash
python main.py
```

### Attributes
* --config
  * Path to the config file. If not provided, the default /config/config.yaml from this project will be used.

* --log-in-file
  * Boolean variable. If set to true, all logs shown in terminal will also be copied in a log file.

### The config file
In order for this pipeline to run, a config file must be provided. It looks something like this:
```yaml
model:
  name: "mlp"                 
  num_classes: 10          

dataset:
  name: "MNIST"            
  path: "./data"
  batch_size: 64
  num_workers: 4
  augmentation: "basic"   
  pin_memory: True

training:
  device: "cuda"            
  epochs: 5
  early_stopping: 5         
  optimizer: "Adam"          
  learning_rate: 0.001
  scheduler: "StepLR"       
  scheduler_params:        
    step_size: 10
    gamma: 0.1
```

#### Variables
* model 
  * name: 
    * mlp
    * lenet
    * resnet18_cifar10
    * preact_resnet18
  * num_classes
    * 10 for mlp, lenet and resnet18_cifar10
    * 100 for preact_resnet18
* dataset:
  * name
    * MNIST for mlp and lenet
    * CIFAR10 for resnet18_cifar10
    * CIFAR100 for preact_resnet18
  * path
    * path to the dataset. If it does not exist, it will be downloaded there
  * batch_size
  * num_workers:
    * unused at the moment
  * augmentation:
    * basic
    * advanced
    * none
  * pin_memory:
    * true
    * false
* training:
  * device
    * cuda
    * cpu
  * epochs
  * early_stopping 
    * set to None to disable early stopping
  * optimizer
  * learning_rate
  * scheduler
    * StepLR
    * ReduceLROnPlateau
    * None
  * scheduler_params:
    * step_size
    * gamma
