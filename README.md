# NumPy

I build a basic deep neural network with 4 layers: 1 input layer, 2 hidden layers, and 1 output layer. All of the layers are fully connected. I'm trying to classify digits from 0 - 9 using a data set called MNIST. This data set consists of 70,000 images that are 28 by 28 pixels each. The data set contains one label for each image that specifies the digit that I see in each image. I say that there are 10 classes because I have 10 labels.
<br><br>
![mist image](img/mnist-1.png)
10 examples of the digits from the MNIST data set, scaled up 2x
<br><br>
For training the neural network, I use stochastic gradient descent, which means I put one image through the neural network at a time.
<br><br>
Let's try to define the layers in an exact way. To be able to classify digits, you must end up with the probabilities of an image belonging to a certain class after running the neural network because then you can quantify how well your neural network performed.
<br>
1. Input layer: In this layer, I input my data set consisting of 28x28 images. I flatten these images into one array with 28×28=78428×28=784 elements. This means that the input layer will have 784 nodes.
<br>
2. Hidden layer 1: In this layer, I reduce the number of nodes from 784 in the input layer to 128 nodes. This creates a challenge when you are going forward in the neural network (I'll explain this later).
<br>
3. Hidden layer 2: In this layer, I decide to go with 64 nodes, from the 128 nodes in the first hidden layer. This is no new challenge because I've already reduced the number in the first layer.
<br>
4. Output layer: In this layer, I reduce the 64 nodes to a total of 10 nodes so that I can evaluate the nodes against the label. This label is received in the form of an array with 10 elements, where one of the elements is 1 while the rest are 0.
<br>

You probably realize that the number of nodes in each layer decreases from 784 nodes to 128 nodes to 64 nodes to 10 nodes. This is based on empirical observations that this yields better results because we're not overfitting nor underfitting, only trying to get just the right number of nodes. The specific number of nodes chosen for this article were chosen at random, although decreasing to avoid overfitting. In most real-life scenarios, you would want to optimize these parameters by brute force or good guesses, usually by grid search or random search, but this is outside the scope of this article.<br>
![mist image](img/deep_nn-1.png)

## The numbers of hidden layers
This is a repost/update of previous content that discussed how to choose the number and structure of hidden layers for a neural network. I first wrote this material during the “pre-deep learning” era of neural networks. Deep neural networks have somewhat changed the more classical recommendations of having at most 2 layers and how to choose the number of hidden layers.

A single hidden layer neural networks is capable of [universal approximation]https://en.wikipedia.org/wiki/Universal_approximation_theorem. The universal approximation theorem states that a feed-forward network, with a single hidden layer, containing a finite number of neurons, can approximate continuous functions with mild assumptions on the activation function. The first version of this theorem was proposed by Cybenko (1989) for sigmoid activation functions. Hornik (1991) expanded upon this by showing that it is not the specific choice of the activation function, but rather the multilayer feedforward architecture itself which allows neural networks the potential of being universal approximators.

Due to this theorem you will see considerable literature that suggests the use of a single hidden layer. This all changed with Hinton, Osindero, & Teh (2006). If a single hidden layer can learn any problem, why did Hinton et. al. invest so heavily in deep learning? Why do we need deep learning at all? While the universal approximation theorem states/proves that a single layer neural network can learn anything, it does not specify how easy it will be for that neural network to actually learn something. Every since the multilayer perceptron, we’ve had the ability to create deep neural networks. We just were not particularly good at training them until Hinton’s groundbreaking research in 2006 and subsequent advances that built upon his seminal work.

Traditionally, neural networks only had three types of layers: hidden, input and output. These are all really the same type of layer if you just consider that input layers are fed from external data (not a previous layer) and output feed data to an external destination (not the next layer). These three layers are now commonly referred to as dense layers. This is because every neuron in this layer is fully connected to the next layer. In the case of the output layer the neurons are just holders, there are no forward connections. Modern neural networks have many additional layer types to deal with. In addition to the classic dense layers, we now also have dropout, convolutional, pooling, and recurrent layers. Dense layers are often intermixed with these other layer types.

This article deals with dense laeyrs. When considering the structure of dense layers, there are really two decisions that must be made regarding these hidden layers: how many hidden layers to actually have in the neural network and how many neurons will be in each of these layers. We will first examine how to determine the number of hidden layers to use with the neural network.

Problems that require more than two hidden layers were rare prior to deep learning. Two or fewer layers will often suffice with simple data sets. However, with complex datasets involving time-series or computer vision, additional layers can be helpful. The following table summarizes the capabilities of several common layer architectures.

##### **Table: Determining the Number of Hidden Layers**


0. none(hidden layer)	Only capable of representing linear separable functions or decisions.
1. (hidden layer)	Can approximate any function that contains a continuous mapping from one finite space to another.
2. (hidden layer)	Can represent an arbitrary decision boundary to arbitrary accuracy with rational activation functions and can approximate any smooth mapping to any accuracy.
3. (greater than 2 hidden layer) Additional layers can learn complex representations (sort of automatic feature engineering) for layer layers.

Deciding the number of hidden neuron layers is only a small part of the problem. You must also determine how many neurons will be in each of these hidden layers. This process is covered in the next section.

## The Number of Neurons in the Hidden Layers

Deciding the number of neurons in the hidden layers is a very important part of deciding your overall neural network architecture. Though these layers do not directly interact with the external environment, they have a tremendous influence on the final output. Both the number of hidden layers and the number of neurons in each of these hidden layers must be carefully considered.

Using too few neurons in the hidden layers will result in something called underfitting. Underfitting occurs when there are too few neurons in the hidden layers to adequately detect the signals in a complicated data set.

Using too many neurons in the hidden layers can result in several problems. First, too many neurons in the hidden layers may result in overfitting. Overfitting occurs when the neural network has so much information processing capacity that the limited amount of information contained in the training set is not enough to train all of the neurons in the hidden layers. A second problem can occur even when the training data is sufficient. An inordinately large number of neurons in the hidden layers can increase the time it takes to train the network. The amount of training time can increase to the point that it is impossible to adequately train the neural network. Obviously, some compromise must be reached between too many and too few neurons in the hidden layers.

I have a few rules of thumb that I use to choose hidden layers. There are many rule-of-thumb methods for determining an acceptable number of neurons to use in the hidden layers, such as the following:

1. The number of hidden neurons should be between the size of the input layer and the size of the output layer.
2. The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
3. The number of hidden neurons should be less than twice the size of the input layer.


These three rules provide a starting point for you to consider. Ultimately, the selection of an architecture for your neural network will come down to trial and error. But what exactly is meant by trial and error? You do not want to start throwing random numbers of layers and neurons at your network. To do so would be very time consuming.


# Imports and data sets
For the entire NumPy part, I specifically wanted to share the imports used. Note that I use libraries other than NumPy to more easily load the data set, but they are not used for any of the actual neural networks.




```python
from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time
```

Now, I must load the data set and preprocess it so that I can use it in NumPy. I do normalization by dividing all images by 255 and make it such that all images have values between 0 - 1 because this removes some of the numerical stability issues with activation functions later on. I use one-hot encoded labels because I can more easily subtract these labels from the output of the neural network. I also choose to load the inputs as flattened arrays of 28 * 28 = 784 elements because that is what the input layer requires.




```python
x,y = fetch_openml('mnist_784',version=1,return_X_y = True)
# x = (x/255).astype('float32') 
# y = to_categorical(y)
# x_train , x_validation , y_train ,y_validation = train_test_split(x,y,test_size = 0.15, random_state=42)
```


```python
x = (x/255).astype('float32') 
y = to_categorical(y)
x_train , x_val , y_train ,y_val = train_test_split(x,y,test_size = 0.15, random_state=42)
```

# Initialization
The initialization of weights in the neural network is a little more difficult to think about. To really understand how and why the following approach works, you need a grasp of linear algebra, specifically dimensionality when using the dot product operation.

The specific problem that arises when trying to implement the feedforward neural network is that we are trying to transform from 784 nodes to 10 nodes. When instantiating the **DeepNeuralNetwork** class, I pass in an array of sizes that defines the number of activations for each layer.



```python
dnn = DeepNeuralNetwork(sizes=[784,128,64,10])
```

This initializes the **DeepNeuralNetwork** class by the **__init__** function.


```python
def __init__(self,sizes,epochs = 10, l_rate = 0.001):
    self.sizes = sizes
    self.epochs = epochs
    self.l_rate = l_rate
    
    # we save all parameters in neural network in this directory
    self.params = self.initialization()
```

Let's look at how the sizes affect the parameters of the neural network when calling the **initialization()** function. I am preparing **m x n** matrices that are **"dot-able"** so that I can do a forward pass, while shrinking the number of activations as the layers increase. I can only use the dot product operation for two matrices **M1** and **M2**, where **m** in **M1** is equal to **n** in **M2**, or where **n** in **M1** is equal to **m** in **M2**.

With this explanation, you can see that I initialize the first set of weights **W1** with **m=128m=128** and **n=784n=784**, while the next weights **W2** are **m=64m=64** and **n=128n=128**. The number of activations in the input layer **A0** is equal to **784**, as explained earlier, and when I dot **W1** by the activations **A0**, the operation is successful.


```python
def initialization(self):
    #number of nodes in each layer 
    input_layer = self.sizes[0]
    hidden_1=self.sizes[1]
    hidden_2= self.sizes[2]
    output_layer=self.sizes[3]
    params= {
        'W1':np.random.randn(hidden_1,input_layer)*np.sqrt(1. / hidden_1),
        'W2':np.random.randn(hidden_2, hidden_1)*np.sqrt(1. / hidden_2),
        'W3':np.random.randn(output_layer,hidden_2)*np.sqrt(1. / output_layer)
    }
    return params
```

# Feedforward

The forward pass consists of the dot operation in NumPy, which turns out to be just matrix multiplication. As described in the **[Introduction to neural networks](https://mlfromscratch.com/neural-networks-explained/)** article, I must multiply the weights by the activations of the previous layer. Then, I must apply the activation function to the outcome.

To get through each layer, I sequentially apply the dot operation followed by the sigmoid activation function. In the last layer, I use the softmax activation function because I want to have probabilities of each class so that I can measure how well the current forward pass performs.

**Note**: I chose a numerically stable version of the softmax function. You can read more from the course at Stanford called **[CS231n](https://cs231n.github.io/linear-classify/#softmax)**.


```python
def forward_pass(self,x_train):
    params = self.params
    
    # input layer activation becomes sample
    params['A0'] = x_train
    
    #input layer to hidden layer 1
    params['Z1'] = np.dot(params["W1"],params['A0'])
    params['A1'] = self.sigmoid(params['Z1'])
    
    ## hidden layer 1 to hidden layer 2
    params['Z2'] = np.dot(params["W2"],params['A1'])
    params['A2']= self.sigmoid(params['Z2'])
    
    ## hideen layer 2 to output layer
    params['Z3'] = np.dot(params["W3"],params['A2'])
    params['A3']= self.softmax(params['Z3'])
    
    return params['A3']

```

#### Activation function
The following code shows the activation functions used for this article. As can be observed, I provide a derivative version of the sigmoid because I need that later on when backpropagating through the neural network.
![activation function](img/activationfunctions.gif)



```python
def sigmoid(self,x,derivative =False):
    if(derivative):
        return (np.exp(-x)/(np.exp(-x)+1)**2)
    return 1/(1+np.exe(-1))
def softmax(self,x):
    #numerically stable with large exponentials
    exps = no.exp(x-x.max())
    return exps/np.sun(exps,axis=0)
```

# Backpropagation
The backward pass is hard to get right because there are so many sizes and operations that must align for all of the operations to be successful. Here is the full function for the backward pass. I go through each weight update below.



```python
def backward_pass(self,y_train,output):
    '''This is the back propagation alorithm, for calculation the updates 
       of the neural networks's parameters
    '''
    params = self.params
    change_w={}
    
    #calculate w3 update
    error = output - y_train
    change_w['W3'] = np.dot(error,params['A3'])
    
    #calculate W2 update 
    error = np.multiply(np.dot(params['W3'].T,error),self.sigmoid(params['Z2'],derivative = True))
    change_w['W2']= np.dot(error,params['A2'])
    
    #  calculate W1 update 
    error = np.multiply(np.dot(params['W2'].T,error),self.sgmoid(params['Z1'],derivative = True))
    change_w['W1']= np.dot(error,params['A1'])
    
    return change_w

```

## W3 update 
The update for W3 can be calculated by subtracting the ground truth array with labels called **y_train** from the output of the forward pass called output. This operation is successful because **len(y_train)** is 10 and **len(output)** is also 10. An example of **y_train** might be the following code, where the 1 is corresponding to the label of the output.
```
y_train = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
```
An example of output is shown in the following code, where the numbers are probabilities corresponding to the classes of **y_train**.
```
output = np.array([0.2, 0.2, 0.5, 0.3, 0.6, 0.4, 0.2, 0.1, 0.3, 0.7])
```
If you subtract them, you get the following.
```
>>> output - y_train
array([ 0.2,  0.2, -0.5,  0.3,  0.6,  0.4,  0.2,  0.1,  0.3,  0.7])
```
The next operation is the dot operation that dots the error (which I just calculated) with the activations of the last layer.
```
error = output - y_train
change_w['W3'] = np.dot(error, params['A3'])
```
## W2 update
Next is updating the weights W2. More operations are involved for success. First, there is a slight mismatch in shapes because W3 has the shape (10, 64) and error has (10, 64), that is, the exact same dimensions. Therefore, I can use a **[transpose operation](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html#numpy.transpose)** on the W3 parameter by the ***.T*** such that the array has its dimensions permuted and the shapes now align up for the dot operation.

![transpose image](img/transpose.png)
An example of the transpose operation. Left: The original matrix. Right: The permuted matrix

**W3** now has shape **(64, 10)** and error has shape (10, 64), which are compatible with the dot operation. The result is **[multiplied element-wise](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)** (also called Hadamard product) with the outcome of the derivative of the sigmoid function of **Z2**. Finally, I dot the error with the activations of the previous layer.
```
error = np.multiply( np.dot(params['W3'].T, error), self.sigmoid(params['Z2'], derivative=True) )
change_w['W2'] = np.dot(error, params['A2'])
```

## W1 update
Likewise, the code for updating **W1** is using the parameters of the neural network one step earlier. Except for other parameters, the code is equivalent to the **W2** update.
```
error = np.multiply( np.dot(params['W2'].T, error), self.sigmoid(params['Z1'], derivative=True) )
change_w['W1'] = np.dot(error, params['A1'])
```

# Training (stochastic gradient descent)
I have defined a forward and backward pass, but how can I start using them? I must make a training loop and use stochastic gradient descent (SGD) as the optimizer to update the parameters of the neural network. There are two main loops in the training function. One loop for the number of epochs, which is the number of times I run through the entire data set, and a second loop for running through each observation one by one.

For each observation, I do a forward pass with x, which is one image in an array with the length 784, as explained earlier. The output of the forward pass is used along with y, which are the one-hot encoded labels (the ground truth) in the backward pass. This gives me a dictionary of updates to the weights in the neural network.




```python
def train(self, x_train, y_train, x_val, y_val):
    start_time = time.time()
    for iteration in range(self.epochs):
        for x,y in zip(x_train, y_train):
            output = self.forward_pass(x)
            changes_to_w = self.backward_pass(y, output)
            self.update_network_parameters(changes_to_w)

        accuracy = self.compute_accuracy(x_val, y_val)
        print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2}'.format(iteration+1, time.time() - start_time, accuracy))
```

The ***update_network_parameters()*** function has the code for the SGD update rule, which just needs the gradients for the weights as input. And to be clear, SGD involves calculating the gradient using backpropagation from the backward pass, not just updating the parameters. They seem separate, and they should be thought of separately because the two algorithms are different.


```python
def update_network_parameters(self, changes_to_w):
    '''
        Update network parameters according to update rule from
        Stochastic Gradient Descent.

        θ = θ - η * ∇J(x, y),
            theta θ:            a network parameter (e.g. a weight w)
            eta η:              the learning rate
            gradient ∇J(x, y):  the gradient of the objective function,
                                i.e. the change for a specific theta θ
    '''

    for key, value in changes_to_w.items():
        for w_arr in self.params[key]:
            w_arr -= self.l_rate * value
```

After having updated the parameters of the neural network, I can measure the accuracy on a validation set that I prepared earlier to validate how well the network performs after each iteration over the whole data set.

The following code uses some of the same pieces as the training function. To start, it does a forward pass then finds the prediction of the network and checks for equality with the label. After that, I sum over the predictions and divide by 100 to find the accuracy. Next, I average out the accuracy of each class.


```python
def compute_accuracy(self, x_val, y_val):
    '''
        This function does a forward pass of x, then checks if the indices
        of the maximum value in the output equals the indices in the label
        y. Then it sums over each prediction and calculates the accuracy.
    '''
    predictions = []

    for x, y in zip(x_val, y_val):
        output = self.forward_pass(x)
        pred = np.argmax(output)
        predictions.append(pred == y)

    summed = sum(pred for pred in predictions) / 100.0
    return np.average(summed)
```

Finally, I can call the training function after knowing what will happen. I use the training and validation data as input to the training function, and then wait.


```python
dnn.train(x_train, y_train, x_val, y_val)
```

Note that the results might vary a lot depending on how the weights are initialized. My results range from an accuracy of 0% - 95%.

Following is the full code for an overview of what's happening.




```python
from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
y = to_categorical(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = output - y_train
        change_w['W3'] = np.dot(error, params['A3'])

        # Calculate W2 update
        error = np.multiply( np.dot(params['W3'].T, error), self.sigmoid(params['Z2'], derivative=True) )
        change_w['W2'] = np.dot(error, params['A2'])

        # Calculate W1 update
        error = np.multiply( np.dot(params['W2'].T, error), self.sigmoid(params['Z1'], derivative=True) )
        change_w['W1'] = np.dot(error, params['A1'])

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y),
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''

        for key, value in changes_to_w.items():
            for w_arr in self.params[key]:
                w_arr -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == y)

        summed = sum(pred for pred in predictions) / 100.0
        return np.average(summed)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2}'.format(iteration+1, time.time() - start_time, accuracy))

dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
dnn.train(x_train, y_train, x_val, y_val)
```

    Epoch: 1, Time Spent: 34.31s, Accuracy: 0.0
    Epoch: 2, Time Spent: 83.36s, Accuracy: 0.0
    Epoch: 3, Time Spent: 143.52s, Accuracy: 0.0
    Epoch: 4, Time Spent: 192.81s, Accuracy: 0.0
    Epoch: 5, Time Spent: 240.96s, Accuracy: 0.0
    Epoch: 6, Time Spent: 292.92s, Accuracy: 0.0
    Epoch: 7, Time Spent: 342.85s, Accuracy: 0.0
    Epoch: 8, Time Spent: 392.40s, Accuracy: 0.0
    Epoch: 9, Time Spent: 442.86s, Accuracy: 0.0
    Epoch: 10, Time Spent: 492.72s, Accuracy: 0.0


# TensorFlow 2.0 with Keras
Now that we know just how much code lies behind a simple neural network in NumPy and PyTorch, let's look at how easily we can construct the same network in TensorFlow (with Keras).

With TensorFlow and Keras, we don't have to think as much about activation functions, optimizers etc., since they are already implemented. On top of this, we will see huge improvements in the time it takes to execute and train a neural network, since the frameworks are completely optimized compared to NumPy.

The following approach goes for a complete Keras solution, without a custom training function or anything very TensorFlow related. Go to the end of my **[TensorFlow 2.0](https://mlfromscratch.com/tensorflow-2/#custom-train-and-test-functions-in-tensorflow-2-0)** tutorial to see what a custom training function looks like.


```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import BinaryCrossentropy
```


```python
(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train = x_train.astype('float32') / 255
y_train = to_categorical(y_train)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 2s 0us/step



```python
model = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(10)
])

model.compile(optimizer='SGD',
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

```



    Epoch 1/10
    1875/1875 [==============================] - 1s 765us/step - loss: 2.4278 - accuracy: 0.1577
    Epoch 2/10
    1875/1875 [==============================] - 1s 784us/step - loss: 4.7711 - accuracy: 0.0904
    Epoch 3/10
    1875/1875 [==============================] - 2s 832us/step - loss: 7.5988 - accuracy: 0.0988
    Epoch 4/10
    1875/1875 [==============================] - 1s 778us/step - loss: 8.8277 - accuracy: 0.0993
    Epoch 5/10
    1875/1875 [==============================] - 1s 776us/step - loss: 8.8277 - accuracy: 0.0993
    Epoch 6/10
    1875/1875 [==============================] - 1s 762us/step - loss: 8.8277 - accuracy: 0.0993
    Epoch 7/10
    1875/1875 [==============================] - 1s 781us/step - loss: 8.8277 - accuracy: 0.0993
    Epoch 8/10
    1875/1875 [==============================] - 1s 763us/step - loss: 8.8277 - accuracy: 0.0993
    Epoch 9/10
    1875/1875 [==============================] - 1s 770us/step - loss: 8.8277 - accuracy: 0.0993
    Epoch 10/10
    1875/1875 [==============================] - 1s 774us/step - loss: 8.8277 - accuracy: 0.0993





    <tensorflow.python.keras.callbacks.History at 0x7f12c51fcd00>



### [Rreference](https://developer.ibm.com/articles/neural-networks-from-scratch/)


```python

```
