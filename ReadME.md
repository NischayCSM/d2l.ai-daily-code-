## Day 1

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

---

## Day 2

### 3.4
* implemented linear regression from scratch by first making a class for linear regression that accepted took number of inputs and learning rate as inputs
* then added two method to the class forward that calculates the output value and loss to calculate the loss
* the made an SGD class i.e. standard gradient descent to calculate the stocastic gradient of minibatch made
* then added a method that calls sgd class to the main linear regression class
* finally added a method to the trainer class made in 3.2 which the model is gone through epochs where the data is extracted for either training or validation
* then finally ran the model and train it and got a plot graph as an output.

---

### 3.5
* while 3.4 focused on how to make linear regression from scratch, this chapter focuses on the easier and faster implementation of linear regression
* used apis for faster implementation and by using apis there is no need for some methods that were in the implementation from scratch.

---

### 3.7
* implement it frim scratch used higher dimensional linear regression data, there defined a l2 penalty function
* then class weight decay where we initialize number of inputs, lambda, learning rate and sigma and then we calculate loss
* then finally trained the model without regularization and with regularization and got the respective graphs as output.