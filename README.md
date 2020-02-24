# Custom deep learning library

We wanted to implement python library that supports working with generic
Artifitial Neural Network (ANN). The point was not to use any framework, but
to dig into the math that is behind the modern Artifitial Intelligence.

## Getting started

```python
model = Model([[4], [2, "relu"], [3, "softmax"]], loss_function='cross_entropy', learning_rate=0.001)
```

When creating model of neural network, user has to provide list of layers (input, hidden and output), including activation functions they want to choose, according to the problem. Next up is to pass loss function type and learning rate as other two parameters.

After the Model has been initialized, it's time to train it using model.train() function. There are 2 types of algorithms for training our network: 
1. Gradient Descent
2. Genetic Algorithm

```python
model.train("gradient_descent", x_train, y_train, epochs=5, shuffle_data=True, batch_size=10)
```

```python
model.train("genetic_algorithm", np.array(training_data), np.array(targets), epochs=5, shuffle_data=True, batch_size=10)
```

After choosing the algorithm for training, required parameters are training data and targeting data. User needs to provide those two data sets, and then the net can be trained. Other parameters are provided by default, but user can change them and track the performance of it's net.

Losses

Activations
