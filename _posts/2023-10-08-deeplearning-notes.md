---
title: "Deep Learning Notes"
published: true
tags: deep-learning DL ML machine-learning python 
sidebar:
  title: "Table of Contents"
  nav: dl-toc
description: "Summary notes of Course 1-3 DeepLearning.ai - 10 min read"
---

# Deep Learning notes
Summary notes from [deeplearning.ai's](https://www.deeplearning.ai/courses/deep-learning-specialization/) Deep Learning Specialization Course 1-3. The maths such as function's equation and derivatives is not included here. Only the first three courses were included as they cover basic neural network as well as very general yet _important_ deep learning framework/methodology that can and _should_ be applied too all machine learning/deep learning projects. 

If you stumble upon this and cannot follow any of this, I do apologize as I collated this for my own personal reference, this is quite an established course with many other learners sharing their thoughts and notes which would better serve you. 

<a class="anchor" id="basic"></a>

## Basics of Neural Network
Neural network (NN) is layers of neurons that maps input to output through hidden layers. There are 3 main sections of NN:

1. Input Layer - takes in the data, number of neurons would usually be number of features of data.
2. Hidden Layer(s) - Number of layers of number of neurons within them are hyperparameters to be tuned/decided.
3. Output Layer - the layer that gives the output.

NN can be thought as linear model on linear model with multiple nodes in a layer passing its output to the next layer, until the output layer.

The neurons within these would have their own activation function (AF) that "transforms" the values of previous layer's. There are various options to AF that includes:

1. Sigmoid Function
2. Tanh Function
3. Rectified Linear Unit (ReLU)
4. Leaky ReLU

The AF of choice does not have to be the same across all layers, each layer can have a different AF from others.

Sigmoid is the least preferred AF except for output layer with binary outcome. ReLU is the most popular option. 

All these AF are non-linear because having a linear function would just result in a linear model, this is because composition of two linear functions itself is a linear function. This would render the hidden layers 'useless'. Therefore non-linear AF in the hidden layer is important. Linear function is only useful for output layer for regression problem.

<a class="anchor" id="input"></a>

## Input data
Two main types of data, structured vs unstructured, structured are tabulated or database data with well-defined meaning. Unstructured data are audio, image and text. 

It is always good to normalize the input data, this is usually done by Standardizing (e.g. [StandardScaler in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)) with mean subtraction and dividing with standard deviation. This step would make it easier for gradient descent to find the minima (converged).

In 'classic' ML, input dataframe are structured as row for samples, columns for features. But in NN, it is inversed, rows are features, columns are samples. 

For ML, to take in image data are represented intensity of pixel value, (in a black shade image). They would be 2D. So they have to be reshaped from 2D into 1D vector of input. Further, in coloured images, it would be three 2D dataset, each one representing the RGB colours, the three of them would have to be stacked as one. Finally, to standardize image data, we divide them with 255 as that's the maximum value it can take. 

<a class="anchor" id="steps"></a>

## Main steps in NN
1. Define that network layers and its nodes count.
2. Initialize values for parameters, which are the weights and biase(random values for weights, zero for bias).
    - weights need to be initialized with random values to break symmetry during optimization.
3. Optimize the parameters by:
    - computing loss/cost with forward propagation
    - obtain derivatives with back-propagation
    - update parameters with gradient descent (or its variants)
4. use learned parameters to predict the outcome. 

<a class="anchor" id="shallow"></a>

### Shallow NN
Shadow NN are NN with only one hidden layer. Input layer is not counted in NN architecture, therefore shallow NN is referred to as 2 layer NN. 

Useful tip to debugging NN, checking the dimensions of weight, where it should have (_n_, _m_) dimension, where _n_ is number of nodes in current layer, _m_ is number of nodes in previous layer. Bias is (_n_, 1) dimension.

<a class="anchor" id="deep"></a>

### Deep NN
Deep NN are NN with more than one hidden layer. If writing the NN by hand, there is no clear way to propagate from layer 1 to layer 2, therefore we need _for_ loop to iterate through the layers. The theory is the same as before, just that we need to iterate through the layers and store the values of the previous layer's output as 'cache' for back-propagation. 

<a class="anchor" id="hyperparameters"></a>

### Hyperparameters
Hyperparameters are parameters that are not learned by the model, but are set by the user. These includes:

1. Number of hidden layers, _L_
2. Number of nodes in each hidden layer, _n_
3. Learning rate, _α_
4. Choice of Activation Function
5. Number of training iterations
6. Choice of optimization algorithm, and their hyperparameters (e.g. momentum, mini-batch size, etc.)
7. Regularization hyperparameters (e.g. L2 regularization parameter, dropout hyperparameters)

Choosing hyperparameters is an art, and there is no clear way to do it. It is usually done by trial and error, and experience.

The order importance for hyperparameters are:

1. Learning rate, _α_
2. Number of hidden units, _n_
3. Optimization algorithm's hyperparameters, e.g. momentum, mini-batch size, etc.
4. Number of layers, _L_
5. Learning rate decay

Basic idea of tuning hyperparameters is to start with random values. Use a coarse to fine approach, where we start with a wide range of values, then narrow down to the best value.

#### Tuning Learning rate, _α_
Learning rate is the most important hyperparameter. If it is too small, it would take a long time to converge. If it is too large, it would overshoot the minima and diverge. It should be ranging from 0.0001 to 1 as a simple starting point. Or anything less than 1. 

#### Tuning momentum, _β_
Momentum is a hyperparameter for optimization algorithm. It is a value between 0 to 1, where 0 means no momentum, and 1 means full momentum. Momentum is used to speed up gradient descent, and it is used to smooth out the gradient descent path. It should range from 0.9 to 0.999.

#### Updating hyperparameters
Over time hyperparameters can get 'stale', this could be due to machine being upgraded, and dataset changing. When this happens it is important to re-tune the hyperparameters.

<a class="anchor" id="practical"></a>

## Practical Aspects of DL
Applied ML is an iterative process, where we would have an idea, build a model, test it, and repeat.

### Train/dev/test distributions
To test/evaluate the models, we need train/dev/test sets. Dev set are also known as holdout or cross-validation set. Small dataset (<2000 samples) would make use of 60/20/20 split, while large data would use 98/1/1 split or 99.5/0.25/0.25 split.

The idea here is to use the dev set to tune the hyperparameters (think develop the model), and use the test set to evaluate the tuned model. Depending on the split, after tuning the hyperparameters, we can retrain the model with the dev set (train+dev set) using the optimal hyperparameters found and evaluate the model with the test set. 

In machine learning, nested K-fold cross-validation (CV) is done, where the data is split into K folds (the outer CV loop), and K-1 folds are used for training, and the remaining fold is used for testing/evaluation. This is repeated K times, where each fold is used as the test set once. Using the outer training set, we can further split it into train and dev set (the inside CV loop), where the train set is used for training, and the dev set is used for tuning the hyperparameters. This is typically computationally intensive, so is NN training, therefore nested CV is not usually done for NN.

It is important to make sure that the data are representative of the population and the dev/test come from the same distribution, and that the data are shuffled before splitting.

### Bias and Variance
High bias indicates underfitting, high variance indicates overfitting.

To fix high bias, we can:
- make NN bigger (more nodes, more layers)
- train longer
- try different NN architecture

To fix high variance, we can:
- get more data
- regularize
- try different NN architecture

<a class="anchor" id="regularize"></a>

### Regularization
Regularization is a technique to reduce overfitting. There are two types of regularization typically used in linear models, L1 and L2. L1 regularization is also known as Lasso regression, while L2 is known as Ridge regression. 

In NN, we use Frobenius norm, which is the sum of squares of all elements in a matrix, a.k.a. L2 regularization is also known as "weight decay". For regularization we need to tune the hyperparameter λ, which is the regularization parameter.

Basic idea of how regularization works is by shrinking the weights, which in turn makes the output smaller of certain nodes smaller and therefore less likely to be "activated". This reduces the complexity of NN and therefore reduces overfitting. Nodes get 'deactivated' when the weights are shrunk to zero, and the weight is associated with the output of the node, zero weight means zero output. Reminder the equation for a node is `z = w*x + b`, where _w_ is the weight, _x_ is the input, _b_ is the bias, _z_ is the output.

### Dropout Regularization
Dropout is a regularization technique that randomly 'drops' some nodes in each iteration. This is done by setting the weights of the nodes to zero. Dropout is dependent on the layer, and the probability of dropping the nodes is a hyperparameter, i.e. different layers have different probability of dropping nodes.

### Other Regularization Methods
Other regularization methods includes:
- Data augmentation, which includes flipping, rotating, cropping, etc. of the images. This is useful for image data.
- Early stopping

<a class="anchor" id="metrics"></a>

### Single number evaluation metric
In machine learning, there's [multiple evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) available to determine how well your model is performing. To simplify and speed up DL training process, it is better to have a single evaluation for a given problem to expedite decision making on next steps for the model. For example with classification, we can use F1 score, which is the harmonic mean of precision and recall.

### Satisficing and Optimizing metric
Satistificing metrics is a metric where as long as it meets a certain threshold, we can be satisfied with. For example, we wants a certain execution speed, as long as it completes execution under that time, we don't have to iterate further to improve on that metrics. Typically stakeholders would set a satisficing metrics.

Optimizing metrics is a metric where we want to have the best possible outcome. For example, we want to improve accuracy as much as possible. Unlike satisficing metrics, we would want to keep improving on this metric. 

Along the previous note on single number evaluation metrics, when we have multiple goals given to us, we can have multiple satisficing metrics, but only one optimizing metric.

<a class="anchor" id="optimize"></a>

## Optimization Algorithms

1. Gradient Descent (The original), which uses all data points to update the parameters. This is slow for large datasets. Best used when the dataset is small (less than 2000 samples).
2. Stochastic Gradient Descent (SGD), which is a variant of GD that uses single data point to update the parameters. But the path to the minima is not smooth.
3. Mini-batch Gradient Descent, which is a variant of GD that uses a batch of data points to update the parameters. This is the most popular method.
    - choosing mini-batch size is a hyperparameter, but it must fit in the CPU/GPU memory. Further, it should be a power of 2, e.g. 64, 128, 256, etc.
4. Momentum, which is a variant of GD that uses the average of gradients to update the parameters. Need to tune the hyperparameter β, which is the momentum parameter.
5. RMSprop, which is a variant of GD that uses the average of squared gradients to update the parameters. Need to tune the hyperparameter β, which is the momentum parameter.
6. Adam, which is a variant of GD that uses the average of gradients and squared gradients to update the parameters (combination of momentum and RMSprop). Need to tune the hyperparameters β1 and β2, which are the momentum parameters. 

<a class="anchor" id="approaches"></a>

## Model Building Approaches

### Pandas vs Caviar
Two approaches to building model, named after the animals' child rearing process. Pandas is the approach of building one model at a time, while Caviar is the approach of building multiple models in parallel. 

### Batch Normalization
Batch normalization is a technique to normalize the output of each layer before feeding to the next layer. This is done by subtracting the mean and dividing by the standard deviation. This is done for each layer, and the mean and standard deviation are calculated for each layer. This is done to speed up training, and it also acts as a regularization technique.


<a class="anchor" id="softmax"></a>

## Softmax Regression
Softmax regression is a generalization of logistic regression to _C_ classes. It is used for multi-class classification. The output of softmax regression is a vector of probabilities, where the sum of the probabilities is 1. DL frameworks such as TensorFlow and PyTorch usually have a built-in function for softmax regression, where it would take the output and transform it into a vector of probabilities.

<a class="anchor" id="strategy"></a>

## ML strategy to improve models
### To fix poor training result
1. Change network architecture
2. Change optimization algorithm
3. Early stopping

### To fix poor dev result
1. Regularization
2. Bigger training set
3. Change network architecture

### To fix poor test result
1. Bigger dev set

### To fix overfitting (poor generalization on real world data)
1. Change dev set
2. Change cost function

## Error Analysis
Error analysis is a technique to determine the cause of error in the model. It is done by manually examining the errors in the dev set, that is pick out all the bad/wrong predictions and go through them one by one to determine what could be the source. For example it could be mislabelled data, or the model is not trained on that type of data. One approach is to have a spreadsheet and columns for the error type, then sum up number of errors for each type. Determine which is the most cost effective to focus on. 

<a class="anchor" id="extension"></a>

## Transfer Learning
Transfer learning is a technique to use a pre-trained model and apply it to a new problem. This is useful when the new problem has a small dataset. The pre-trained model can be trained on a large dataset, and the weights can be used as a starting point for the new problem. For example, NN trained to classify cat images, we can take parts of the knowledge to read X-ray images. 

### How to apply transfer learning
1. Take the pre-trained model, and remove the last layer, or last two layers.
2. Add a new layer with the number of classes of the new problem.
3. Train the new layer on the new dataset. This is also known as fine-tuning.

## Multi-task Learning
Multi-task learning is a technique to train a single NN to perform multiple tasks. This is useful when the tasks have similar features. For example, computer vision, when we want to classify multiple objects in an image. This is not to be confused with softmax regression. 

## End-to-end Deep Learning
End-to-end deep learning is a technique to train a single NN to perform the entire task which usually have a series of steps. For example in audio to text, the traditional approach would be to have a NN to convert audio to text, then another NN to convert text to meaning, as well as other possible steps in between. But with end-to-end deep learning, we can train a single NN to convert audio to meaning. This is useful when there is sufficient data, i.e. it may need lots of data. But this lets the data do the work for us without having to engineer the steps in between. Though this is not always the optimal solution for all the problems, sometimes it is still good to have the traditional approach of having series of steps/models.
