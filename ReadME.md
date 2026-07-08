## Day 1

### 3.1
* understood the formula for linear model, its loss function and the its optimization formula.
* implemented the vectorization for speed to see how much time my cpu will take to add two tensors.
* learnt about normal distribution and implemented it in python and displayed it plot

---

### 3.2
* learnt how to design the model , train the model and store data in object orientation
* learnt about the ultities like add to class and how it works by implemeting it, similarly for progress board
* implemented the class module that templets of initialization, loss calculation, to procees forward in the network, to plot the output, the training and validation steps and finally optimization
* then datamodule class it had it had initialization of the data for the files, to load the data to read , train the loaded data and then validate it
* finally for the trainer class it had initization of the number of epochs, number of gpus and gradient value, it also had the function to prepare the data for training and then to fit the data in the model and finally to fit the epochs

---

### 3.3
* since there no dataset presently we can make one by generating it by a linear equation
* the class SyntheticRegressionData does this calculation if we give it the inputs of weights and bias
* to read the data add a function called get_dataloader to the class, it first checks if the data is for training or for validation
* if for training it creates a list of indices of the data and shuffles them and if validation its takes the list of indices and make them into batches and gives batch indices
* there is also a concise implementation of the synthetic regression data, that does the same as above

---

## Day 2

### 3.4
* implemented linear regression from scratch by first making a class for linear regression that accepted took number of inputs and learning rate as inputs
* then added two method to the class forward that calculates the output value and loss to calculate the loss
* the made an SGD class i.e. standard gradient descent to calculate the stocastic gradient of minibatch made
* then added a method that calls sgd class to the main linear regression class
* finally added a method to the trainer class made in 3.2 which the model is gone through epochs where the data is extracted for either training or validation
* then finally ran the model and train it and got a plot graph as an output

---

### 3.5
* while 3.4 focused on how to make linear regression from scratch, this chapter focuses on the easier and faster implementation of linear regression
* used apis for faster implementation and by using apis there is no need for some methods that were in the implementation from scratch

---

### 3.7
* implement it frim scratch used higher dimensional linear regression data, there defined a l2 penalty function
* then class weight decay where we initialize number of inputs, lambda, learning rate and sigma and then we calculate loss
* then finally trained the model without regularization and with regularization and got the respective graphs as output

---

## Day 3

### 4.2
* implemented an image classification model using fashion mnist data set
* well in general mnist data is a data of handwritten images of number 1 to 9, it is a classic data set used but as time flew it became weak as models became stronger, so as a replacement we use fashion mnist
* this data set has 10 categories of cloths which are " t-shirts, trousers, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boots"
* the model separates the data set and prints a selected image from each category

---

### 4.3
* made an classifier class, this class acts as a base classification class template
* it has a method for validation, optimizer using sgd, and accuracy to tell how good the model is
* since this is a base class there is no output

---

### 4.4
* implemented softmax classification from scratch
* made a softmax method that converts all the values in the tensor to probability which is used for classification
* then made a softmax regression class which inherits classifier base class 
* initialize number of inputs, number of outputs and learning rate, then make parameters of weights and bias
* the in forward function uses the normal linear regression formula but with softmax method wrapped over it
* then compute the loss using cross entropy loss method which i defined
* after that trained the model and got a graph as output, and also i visualized which predictions went wrong

---

### 4.5
* here it is also the same as in 4.4 but instead using formulas uses predefined methods in torch
* for regression in softmax regression class uses lazylinear, and for loss is used the F.crossentropy function defined in torch
* the output is the same as 4.4

---

### 5.1
* implemented relu, sigmoid and tanh function
* plotted relu function and gradient of relu
* plotted sigmoid function and gradient of sigmoid
* plotted tanh function and gradient of tanh

---

## Day 4

### 5.2
*  implemented mlp from scratch
* made a class to initialize the weights and bias of the mlp for the 2-layer mlp
* then made a forward function that calculates values in the 2-layer mlp using linear equations
* wrapped the first layer in relu activation
* then also made a concise implementation using built in modules in pytorch, the output was the same.

---

### 5.4
* implemented code for vanishing gradients and got output
* implemented code for expoding gradients and got output

---

### 5.6
* implement dropout in two way from scratch and concise
* in scratch, made a multilayer perceptron using basicing linear model and then in forward function while executing droped out a few using the dropout function i made
* in concise, did the same but used built in modules in pytorch using a sequential model of a few LazyLinear layer and keeping dropout between them, the output remained the same.

---

### 5.7
*  executed a house price prediction model, and got the data from kaggle
* the data is quite board consisting of nearly 3000 rows of data, and each consisting of nearly 80 columns, this data is downloaded from the url and is stored at location stored in self.root
* then preprocess the data to avoid error during training
* then load the data to the model depending on if it is for training or validation
* for validation k-fold cross validation is used for which k-fold data is made by one function and another calculates the validation
* then the model is trained on the data and output is produced.

---

### 6.1
* here to see how layer and modules worked, implemented a code of a network i multiple ways
* first executed it in a standard fast way using nn.sequential and implemented modules in it
* next  custom made a mlp class of the same layer as in the sequential
* lastly made a custom sequential and implemented the modules in it, all gave the same output shape
* next for implementation of layer in forward propagation , made a hidden mlp that calculates using linear equation and made a nestmlp that has a sequential of lazylinear and relu and after that another lazy linear
* and all this is used as a module in a sequential in the order of nestmlp, lazylinear, fixedhiddenmlp.

---

### 6.2
* for accessing parameter targeted parameter can be accessed by indexing the parameter in the model and using bias.data
* for all parameters a for loop is enough
* for sharing parameter using shared layer in the sequential.