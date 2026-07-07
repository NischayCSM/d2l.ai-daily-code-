import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# Relu activation function
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

plt.savefig('relu-activation-function.png', dpi=300)

y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

plt.savefig('grad-of-relu.png', dpi=300)

#Sigmoid activation function
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

plt.savefig('sigmoid-activation-function.png', dpi=300)

x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

plt.savefig('grad-of-sigmoid.png', dpi=300)

#Tanh activation function
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

plt.savefig('tanh-activation-function.png', dpi=300)

x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

plt.savefig('grad-of-tanh.png', dpi=300)