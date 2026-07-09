import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

print(net[0].weight)

X = torch.rand(2, 20)
net(X)

print(net[0].weight.shape)

@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)

def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

class MyModel(d2l.Module):
    def __init__(self):
        super().__init__()
        # Define your network here
        self.net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
        
    def forward(self, X):
        return self.net(X)
    
# Instantiate the model
model = MyModel()

# Create dummy input data
X = torch.rand(2, 20)

# Pass the input as a tuple (X,) and your init function
model.apply_init((X,), init=init_normal)

# Verify the weights have been initialized
print(model.net[0].weight.shape)