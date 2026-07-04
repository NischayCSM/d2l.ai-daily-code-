# %matplotlib inline
import random
import torch
from d2l import torch as d2l

# ​‌‌​‌‌‍Creating Data​
#SyntheticRegressionData is a class that generates synthetic data, it take weights, bais as inputs for its equation.
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    # the init method here initializes the paramter like weight, bias, noise, number of training and validation data and batch size.
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val # the total number of data points
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise

data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)

print('features:', data.X[0],'\nlabel:', data.y[0])

# ​‌‌‍Reading Data​


@d2l.add_to_class(SyntheticRegressionData)
# the function get_dataloader is used to read the synthetic data to a model.
def get_dataloader(self, train):
    if train: # if the model is used for training the indices list would be made and then it would be shuffled 
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:# for validation the indices list would be split into batches with there batch indices 
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        batch_indices = torch.tensor(indices[i: i+self.batch_size])
        yield self.X[batch_indices], self.y[batch_indices]

X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)

# ​‌‌‍Concise Synthetic Data​

@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, self.batch_size,shuffle=train)

@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)

X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)

len(data.train_dataloader())