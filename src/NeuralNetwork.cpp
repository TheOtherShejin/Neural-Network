#include "NeuralNetwork.h"

namespace ActivationFunctions {
	double ReLU(double input) {
		return fmax(input, 0);
	}
	double Sigmoid(double input) {
		return 1.0f / (1 + exp(-input));
	}
	double HyperbolicTangent(double input) {
		return tanh(input);
	}
}

Layer::Layer(int numOfNodes, int numOfIncomingNodes, double (*activationFunction)(double)) 
	: numOfNodes(numOfNodes), numOfIncomingNodes(numOfIncomingNodes), activationFunction(activationFunction) {
	weights = std::vector<std::vector<double>>(numOfNodes, std::vector<double>(numOfIncomingNodes, 0));
	srand(time(0));
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < numOfNodes; j++) {
			weights[i][j] = rand();
		}
	}

	biases = std::vector<double>(numOfNodes, 0);
	for (int i = 0; i < numOfNodes; i++) {
		biases[i] = rand();
	}
}

void Layer::SetActivationFunction(double (*activationFunction)(double)) {
	this->activationFunction = activationFunction;
}

std::vector<double> Layer::FeedForward(std::vector<double>& input) {
	std::vector<double> output(numOfNodes, 0);

	for (int i = 0; i < numOfNodes; i++) {
		output[i] = biases[i];
		for (int j = 0; j < numOfIncomingNodes; j++) {
			output[i] += input[j] * weights[i][j];
		}
		output[i] = activationFunction(output[i]);
	}

	return output;
}

NeuralNetwork::NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double)) {
	layers.push_back(Layer(numberOfNeurons[0], 0, nullptr));
	for (int i = 1; i < numberOfNeurons.size(); i++) {
		layers.push_back(Layer(numberOfNeurons[i], numberOfNeurons[i - 1], hiddenLayerAF));
	}
	layers[numberOfNeurons.size()-1].SetActivationFunction(outputLayerAF);
}