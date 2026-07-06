import time
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# ​‌‌‌Utlities​
#​‌‌‍ add class
# ​#Function take an object as input , applys setattr to add an object to the class as a method and returns the wrapper function.
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj): 
        setattr(Class, obj.__name__, obj)
    return wrapper

class A:
    def __init__(self):
        self.b = 1

a = A()

@add_to_class(A) #the demo of the add_to_class decorator, it adds the do method to the class A. 
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
# ​‌‌‍Hyperparameter​
#the hyperparameter class is a base class for classes that have a hyperparameter.
class HyperParameters:  #@save 
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]): # this function saves the hyperparameter of that class, the list of ignored is used to ignore those hyperparameters
        raise NotImplemented
    
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):  # the demo of the HyperParameters class, it saves the hyperparameter a and b but ignores c.
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)


#  ​‌‌‍Progress board​
# Progress board is a class made to plot the data ponts in animation
class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""
    # the init function initializes the progress board with the given parameters like xlabel, ylabel, xlim, ylim, xscale, yscale, ls, colors, fig, axes, figsize and display.
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()
    # the draw method is used to draw the data points, its take x,y,label and every_n as inputs, this is a template method and its will be implemented in the child class.
    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
    
board = d2l.ProgressBoard('x') # the demo of the ProgressBoard class.
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)

plt.savefig('day1.1.png')

#  ​‌‌‍Module​
# Module is a base class for all the models, it has methods like loss, forward, plot, training_step, validation_step and configure_optimizers which are to be implemented in the child class.
class Module(nn.Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y): 
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, value.to(d2l.cpu()).detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
    

#  ​‌‌‍DataModule​
# DataModule is a base class for all the data modules, it has methods like get_dataloader, train_dataloader and val_dataloader which are to be implemented in the child class.
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    

# ​‌‌‍Trainer  ​ 
# Trainer is a base class for training models with data, it has methods like prepare_data, prepare_model, fit and fit_epoch which are to be implemented in the child class.
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
    
#There is no output for module, datamodule and trainer classes , they are templete class for creating new classes, they dont give output.