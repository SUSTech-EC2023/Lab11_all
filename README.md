# Lab11
## XOR Problem Solver using Evolutionary Learning Algorithm

This Python script demonstrates the use of an Evolutionary Learning Algorithm to optimize the weights of a feedforward neural network for solving the XOR problem.


## Implementation Details

### Activation Function

The script uses a sigmoid activation function:

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
### Neural Network
The neural network contains two layers:

First layer: 2 input neurons, 3 hidden neurons;

Second layer: 3 hidden neurons, 1 output neuron;

Weights are represented as a flat NumPy array, which is reshaped when passed to the predict function:
```python
def predict(weights, inputs):
    layer1_weights = weights[:6].reshape(2, 3)
    layer2_weights = weights[6:].reshape(3, 1)

    layer1 = sigmoid(np.dot(inputs, layer1_weights))
    output = sigmoid(np.dot(layer1, layer2_weights))

    return output
```

### Genetic Algorithm
A simple Genetic Algorithm is implemented to optimize the weights of the neural network for solving the XOR problem. The inputs and outputs for the XOR problem are provided:
```python
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([[0], [1], [1], [0]])
```
The script also visualizes the progress of the best fitness score per generation using a line plot.
After running the script, you should see the plot, as well as the predictions for the XOR problem using the optimized weights.

**Note:** Please be aware that this script serves as an example of using an evolutionary learning algorithm to solve the XOR problem, and it may not represent the best performance or optimal solution.

