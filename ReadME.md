##Day 1

### 3.1
* understood the formula for linear model, its loss function and the its optimization formula.
* implemented the vectorization for speed to see how much time my cpu will take to add two tensors.
* learnt about normal distribution and implemented it in python and displayed it plot.

---

### 3.2
* learnt how to design the model , train the model and store data in object orientation
* learnt about the ultities like add to class and how it works by implemeting it, similarly for progress board
* implemented the class module that templets of initialization, loss calculation, to procees forward in the network, to plot the output, the training and validation steps and finally optimization
* then datamodule class it had it had initialization of the data for the files, to load the data to read , train the loaded data and then validate it
* finally for the trainer class it had initization of the number of epochs, number of gpus and gradient value, it also had the function to prepare the data for training and then to fit the data in the model and finally to fit the epochs.

---

### 3.3
* since there no dataset presently we can make one by generating it by a linear equation
* the class SyntheticRegressionData does this calculation if we give it the inputs of weights and bias
* to read the data add a function called get_dataloader to the class, it first checks if the data is for training or for validation
* if for training it creates a list of indices of the data and shuffles them and if validation its takes the list of indices and make them into batches and gives batch indices
* there is also a concise implementation of the synthetic regression data, that does the same as above.