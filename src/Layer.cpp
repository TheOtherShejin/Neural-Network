#include "Layer.h"

Layer::Layer(int numOfNodes, int numOfIncomingNodes, double (*activationFunction)(double))
	: numOfNodes(numOfNodes), numOfIncomingNodes(numOfIncomingNodes), activationFunction(activationFunction) {
	weights = std::vector<std::vector<double>>(numOfNodes, std::vector<double>(numOfIncomingNodes, 0));
	weightCostGradients = weights;
	biases = std::vector<double>(numOfNodes, 0);
	biasCostGradients = biases;
	srand(time(0));
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weights[i][j] = ((double)(rand() % 10000) - 5000.0) / 10000.0;
		}
		biases[i] = ((double)(rand() % 10000) - 5000.0) / 10000.0;
	}
}

std::vector<double> Layer::FeedForward(std::vector<double> input) {
	std::vector<double> output(numOfNodes, 0);
	weightedInputs = output;
	rawInputs = input;

	for (int i = 0; i < numOfNodes; i++) {
		weightedInputs[i] = biases[i];
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weightedInputs[i] += input[j] * weights[i][j];
		}

		output[i] = activationFunction(weightedInputs[i]);
	}
	return output;
}

std::vector<double> Layer::CalculateOutputLayerNodeValues(std::vector<double> actualOutput, std::vector<double> expectedOutput) {
	std::vector<double> nodeValues;
	for (int i = 0; i < numOfNodes; i++) {
		nodeValues.push_back(LossDerivative(actualOutput[i], expectedOutput[i]));
		nodeValues[i] *= AF::DerivativeOf(weightedInputs[i], activationFunction);
	}

	return nodeValues;
}

void Layer::ApplyGradients(double learningRate) {
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weights[i][j] -= weightCostGradients[i][j] * learningRate;
		}
		biases[i] -= biasCostGradients[i] * learningRate;
	}
}

void Layer::UpdateGradients(std::vector<double> nodeValues) {
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weightCostGradients[i][j] += rawInputs[j] * nodeValues[i];
		}

		biasCostGradients[i] += nodeValues[i];
	}
}

void Layer::ClearGradients() {
	weightCostGradients = std::vector<std::vector<double>>(numOfNodes, std::vector<double>(numOfIncomingNodes, 0));
	biasCostGradients = std::vector<double>(numOfNodes, 0);
}

std::vector<double> Layer::CalculateHiddenLayerNodeValues(Layer& oldLayer, std::vector<double> oldNodeValues) {
	std::vector<double> newNodeValues(numOfNodes, 0);

	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < oldNodeValues.size(); j++) {
			newNodeValues[i] += oldNodeValues[j] * oldLayer.weights[j][i];
		}
		newNodeValues[i] *= AF::DerivativeOf(weightedInputs[i], activationFunction);
	}

	return newNodeValues;
}

double Layer::LossDerivative(double actualOutput, double expectedOutput) {
	return 2 * (actualOutput - expectedOutput);
}