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

---

## Day 5

### 6.3
*  implemented the parameter initialization learnt in 5.4
* there are 2 types of parameter initialization, there is normal initialization is calculated by means and standard deviation or a constant number
* the second is xavier initialization which has a formula to calculate the parameters
* for these two types of parameters there are built-in functions.

---

### 6.4
* implemented lazy initialization
* made a normal sequential model with lazylinear and provided on parameter initialization , so when printed the weights of the model got <uninitialized parameter>
* then passed the data to model and checked the weight shape it had a size so data got initialized by data input
* for much more detail, made a function called apply_init, for this is needed to make a init function and a class with no init, and used apply_init function to apply the init func to the class.

---

### 6.5
* implemented two layer in the chapter, one with parameters, and other without parameters
* in the one with parameters ,made a class model with weights and bias initialized and the progression model is linear, the weights of the model can be found and send data to see the output
* since it is a parameterized model can have a sequential run on it twice or more
* in the unparameterized model, progression formula is x-mean(x), there is not weights or bias, the model will still give output
* for this model in sequential ,used a predefined model like lazylinear for a more complex model.

---

### 6.6
* this chapter focused more on save data and models
* the first few codes in the chapter was to normal data variables, like vectors, matrices and dict and read them after they are saved 
* the second half is to save a model and its parameters.

---

### 6.7
* this chapter focused on gpus and how to integrate them into the code
* cuda means gpu, here first go through how to access a gpu, then numbering gpus, tensors on gpus, copying in gpus, storing in gpus, and to a run a model in gpu.

---

### 7.2
* first made a function for 2d cross-correlation, then made a class for convolution 2d 
* here the weights and bias are initialized and for progression of the model, cross-correlation of weights and input matrices are used
* for edge detection a tensor of -1 to 1 is made and its cross-correlation is calculated
* then a conv2d model is initialized and x and y tensors are trained on it, the loss is calculated for every epoch it decreases

---

### 7.3
* when convolution is used the size of the image tends to decrease as the formula is (size of image-size of kernel +1), so to prevent the size from decreasing padding is used
* and its implementation is also done in this chapter
* stride is the number of rows or columns traversed by the kernel, by increasing the stride the traversal becomes fast, and computing becomes efficient.

---

### 7.4
* sometime the input data has more that a single channel of data, so to ease out the calculations, the model can loop through the channels and compute
* and other times there is a need for more that one channel of output, so a stack is used to store multiple channel output data.

---

### 7.5
* in max pooling the maximum value of specific input window is taken, for average pooling the average of the input window is taken
* here i made a function that gives output based on if it max pooling or average pooling
* there is also application of padding and stride here and also application of pooling in multiple channels.

---

### 7.6
* LeNet is an old image detection model, it has abouts 2, 5x5 convolution layer and 2 3x3 average pooling layers
* made a lenet class that has conv2d, then average pooling, then conv2d, then average pooling, then flatten, 3 lazy linear layer of which 2 are of sigmoid, so a max of 7 layers
* also made a function to summaries these layer, and finally trained the mode to get the output.

---

## Day 6

### 8.1
* in this chapter. the focus was about deep convolutional networks
* the first deep convolution model was AlexNet, AlexNet is a model 8 CNN layers, this is a model similar to LeNet, but it is much stronger as it has more layer than it
* I also implemented AlexNet , i made a class and implemented modules of relu, convolution 2d , max pooling 2d and lazy linear, then i trained the model on FashionMNIST dataset.

---

### 8.2
* this chapter focuses on network making using blocks, the main examples demonstrated here is VGG
* the VGG block is simple it has 3 types of layer a convolution layer, relu and maxing pooling depending on how many convolutions are need the block expands
* in the main class the blocks are used to compute and then lazy linears are used, this is to make the model a bit flexible and more sophisticated
* the model is trained on FashionMNIST data and output is given.

---

### 8.3
* here it is focused on network in network model
* here a block is made which contains a sequential module having 2d convolutions, then these blocks are used in a main network with maxpooling after each block
* since this is a model that involves two layer of networks it is called network in network
* this type of model is each to control and easy to manipulate, the model is trained on FashionMNIST data and output is given.

---

### 8.4
* it was about multibranch networks, the example of the model used is GoogLeNet
* this model uses a unique approach that is a combination of both block and nin, here there is something called a inception block
* here there are 4 branches 1st branch only has a 1x1 convolution layer, the second branch has a 1x1 convo, then a 3x3 convo, the third branch has a 1x1 convo and 5x5 convo, the last branch has 3x3 max pool and 1x1 convo
* so by using these blocks we add batch function it the main code each has a specific layer limit, and there are a max of 5 batch and one init function
* the model is trained on FashionMNIST data and output is given.

---

### 8.5
* it the concept was batch normalization
* i implemented both from scratch and concise, in scratch i made a function which calculates the batch normalization, then made a class for applying it to neural network in the class
* in concise i used build-in functions in torch for batch normalization, the model is trained on FashionMNIST data and output is given.

---

### 8.6
* this chapter we focused ResNet and ResNeXt
* here to implement the model using something called residual blocks, residual block has convolution layer and batch normalization
* the resnet class convolution layer, max pooling and relu, the also i added a block function here i used residual class to make residual blocks , then i made resnet18 inherits from resnet, which is trained, the same process is used for resnext block
* the model is trained on FashionMNIST data and output is given.

---

### 8.7
* this chapter we focus on DenseNet model
* here i made a dense block class which uses a convolution block
* then i made a transition block, which is used in the main densenet class as a transition between Dense blocks
* the model is trained on FashionMNIST data and output is given.

---

### 8.8
* The last unit is about designing of convolutional models, the example used is AnyNet, which is a model that can be customized
* it has 3 functions, stem which uses sequential model, second is stage which uses ResNeXt blocks, and the last in initialization function
* this model is executed with RegNet model which uses AnyNet as a base design, , the model is trained on FashionMNIST data and output is given.