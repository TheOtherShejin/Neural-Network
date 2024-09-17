A quick and easy way to create Neural Networks in C++

# How To Use

```cpp
NeuralNetwork nn({2, 3, 1}, ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid);
```
The above line of code creates a 3 layer neural network model.
The input layer consists of 2 neurons, the model consists of 1 hidden layer 3 with neurons and an output layer with 1 neuron.
```cpp
nn.Learn({1, 0}, {1}, 1.5);
```
The `NeuralNetwork::Learn()` function takes in the input data (1st parameter), expected output data (2nd parameter) and a learning rate (3rd parameter).
```cpp
nn.CalculateOutput({})
```
The `NeuralNetwork::CalculateOutput()` takes in an input data, and then the neural network does it's thing and spits out an output as an `std::vector<double>`
