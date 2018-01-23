# Deep Learning with Python

![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Keras_Logo.jpg/180px-Keras_Logo.jpg)

Implementation of Deep Learning Algorithms using Keras Library. A large fraction of code examples is based on [François Chollet's Deep Learning with Python Book](https://www.manning.com/books/deep-learning-with-python) and [Jason Brownlee's blog - Machine Learning Mastery](https://machinelearningmastery.com/blog/).


## Table of Contents

1. [A first look at a neural network](https://github.com/andersy005/deep-learning/blob/master/01-A-first-look-at-a-neural-network.ipynb)
2. [A binary classification example: Classifying movie reviews](https://github.com/andersy005/deep-learning/blob/master/02-A-Binary-Classification-Example.ipynb)
3. [A multi classification example: Classifying newswires](https://github.com/andersy005/deep-learning/blob/master/03-A-Multiclass-Classification-Example.ipynb)
4. [A scalar regression example: Predicting house prices](https://github.com/andersy005/deep-learning/blob/master/04-A-Regression-Example-Predicting-House-Prices.ipynb)
5. [Introduction to Convolutional Neural Networks](https://github.com/andersy005/deep-learning/blob/master/05-Introdcution-to-convnets.ipynb)
6. [Dogs vs Cats Classification Using Convnets](https://github.com/andersy005/deep-learning/blob/master/06-CatdogNet.ipynb)
7. [Visualizing what convnets learn](https://github.com/andersy005/deep-learning/blob/master/07-visualizing-what-convnets-learn.ipynb)
8. [Working with text data](https://github.com/andersy005/deep-learning/blob/master/08-working-with-text-data.ipynb) 
9. [Understanding RNNs](https://github.com/andersy005/deep-learning/blob/master/09-understanding-recurrent-neural-networks.ipynb)
10. [Advanced Use of RNNs](https://github.com/andersy005/deep-learning/blob/master/10-advanced-use-of-rnn.ipynb)
11. [Sequence Processing with 1D convnets](https://github.com/andersy005/deep-learning/blob/master/11-sequence-processing-with-convnets.ipynb.ipynb)



## The universal workflow of machine learning

A universal blueprint that you can use to attack and solve any machine-learning problem.

### 1. Defining the problem and assembling a dataset

First, you must define the problem at hand:

- What will your input data be? What are you trying to predict? 
- What type of problem are you facing? 
  - Is it binary classification? 
  - Multiclass classification? 
  - Scalar regression? 
  - Vector regression? 
  - Multiclass, multilabel classification? 
  - Something else, like clustering, generation, or reinforcement learning? 
  
Identifying the problem type will guide your choice of model architecture, loss function, and so on.

You can’t move to the next stage until you know what your inputs and outputs are, and what data you’ll use. Be aware of the hypotheses you make at this stage:

- You hypothesize that your outputs can be predicted given your inputs.
- You hypothesize that your available data is sufficiently informative to learn the relationship between inputs and outputs.

**Be aware of nonstationary problems.**

### 2. Choosing a measure of success

To achieve success, you must define what you mean by success:
 - accuracy? 
 - Precision and recall? 
 - Customer-retention rate? 

**For balanced-classification problems, where every class is equally likely, accuracy and area under the receiver operating characteristic curve (ROC AUC) are common metrics.** 

**For class-imbalanced problems, you can use precision and recall. For ranking problems or multilabel classification, you can use mean average precision.** 


The data science competitions on Kaggle (https://kaggle.com) should be your number one learning resource.

### 3. Deciding on an evaluation protocol

Once you know what you’re aiming for, you must establish how you’ll measure your current progress. We’ve previously reviewed three common evaluation protocols:

- Maintaining a hold-out validation set— The way to go when you have plenty of data
- Doing K-fold cross-validation— The right choice when you have too few samples for hold-out validation to be reliable
- Doing iterated K-fold validation— For performing highly accurate model evaluation when little data is available

Just pick one of these. In most cases, the first will work well enough.

### 4. Preparing your data

Once you know what you’re training on, what you’re optimizing for, and how to evaluate your approach, you’re almost ready to begin training models. But first, you should format your data in a way that can be fed into a machine-learning model—here, we’ll assume a deep neural network:

- Your data should be formatted as tensors.
- The values taken by these tensors should usually be scaled to small values: for example, in the [-1, 1] range or [0, 1] range.
- If different features take values in different ranges (heterogeneous data), then the data should be normalized.
- You may want to do some feature engineering, especially for small-data problems.

Once your tensors of input data and target data are ready, you can begin to train models.


### 5. Developing a model that does better than a baseline

Your goal at this stage is to achieve statistical power: that is, to develop a small model that is capable of beating a dumb baseline. In the MNIST digit-classification example, anything that achieves an accuracy greater than 0.1 can be said to have statistical power; in the IMDB example, it’s anything with an accuracy greater than 0.5.

Note that it’s not always possible to achieve statistical power. If you can’t beat a random baseline after trying multiple reasonable architectures, it may be that the answer to the question you’re asking isn’t present in the input data. Remember that you make two hypotheses:

    You hypothesize that your outputs can be predicted given your inputs.
    You hypothesize that the available data is sufficiently informative to learn the relationship between inputs and outputs.

It may well be that these hypotheses are false, in which case you must go back to the drawing board.

Assuming that things go well, you need to make three key choices to build your first working model:

- **Last-layer activation**— This establishes useful constraints on the network’s output. For instance, the IMDB classification example used sigmoid in the last layer; the regression example didn’t use any last-layer activation; and so on.
- **Loss function**— This should match the type of problem you’re trying to solve. For instance, the IMDB example used binary_crossentropy, the regression example used mse, and so on.
- **Optimization configuration**— What optimizer will you use? What will its learning rate be? In most cases, it’s safe to go with rmsprop and its default learning rate.

| Problem  type                           | Last-layer activation | Loss function              |
|-----------------------------------------|-----------------------|----------------------------|
| Binary classification                   | sigmoid               | binary_crossentropy        |
| Multiclass, single-label classification | softmax               | categorical_crossentropy   |
| Multiclass, multilabel classification   | sigmoid               | binary_crossentropy        |
| Regression to arbitrary values          | None                  | mse                        |
| Regression to values between 0 and 1    | sigmoid               | mse or binary_crossentropy |


### 6. Scaling up: developing a model that overfits

Once you’ve obtained a model that has statistical power, the question becomes, is your model sufficiently powerful? Does it have enough layers and parameters to properly model the problem at hand? For instance, a network with a single hidden layer with two units would have statistical power on MNIST but wouldn’t be sufficient to solve the problem well. Remember that the universal tension in machine learning is between optimization and generalization; the ideal model is one that stands right at the border between underfitting and overfitting; between undercapacity and overcapacity. To figure out where this border lies, first you must cross it.

To figure out how big a model you’ll need, you must develop a model that overfits. This is fairly easy:

    Add layers.
    Make the layers bigger.
    Train for more epochs.

Always monitor the training loss and validation loss, as well as the training and validation values for any metrics you care about. When you see that the model’s performance on the validation data begins to degrade, you’ve achieved overfitting.

The next stage is to start regularizing and tuning the model, to get as close as possible to the ideal model that neither underfits nor overfits.

### 7. Regularizing your model and tuning your hyperparameters

This step will take the most time: you’ll repeatedly modify your model, train it, evaluate on your validation data (not the test data, at this point), modify it again, and repeat, until the model is as good as it can get. These are some things you should try:

    Add dropout.
    Try different architectures: add or remove layers.
    Add L1 and/or L2 regularization.
    Try different hyperparameters (such as the number of units per layer or the learning rate of the optimizer) to find the optimal configuration.
    Optionally, iterate on feature engineering: add new features, or remove features that don’t seem to be informative.

Be mindful of the following: every time you use feedback from your validation process to tune your model, you leak information about the validation process into the model. Repeated just a few times, this is innocuous; but done systematically over many iterations, it will eventually cause your model to overfit to the validation process (even though no model is directly trained on any of the validation data). This makes the evaluation process less reliable.

Once you’ve developed a satisfactory model configuration, you can train your final production model on all the available data (training and validation) and evaluate it one last time on the test set. If it turns out that performance on the test set is significantly worse than the performance measured on the validation data, this may mean either that your validation procedure wasn’t reliable after all, or that you began overfitting to the validation data while tuning the parameters of the model. In this case, you may want to switch to a more reliable evaluation protocol (such as iterated K-fold validation).
