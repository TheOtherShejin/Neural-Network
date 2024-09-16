#include "NeuralNetwork.h"

namespace ActivationFunctions {
	double Linear(double input) {
		return input;
	}
	double BinaryStep(double input) {
		return (input >= 0);
	}
	double ReLU(double input) {
		return fmax(0, input);
	}
	double LeakyReLU(double input) {
		return fmax(0.1f * input, input);
	}
	double Sigmoid(double input) {
		return 1.0f / (1 + exp(-input));
	}
	double TanH(double input) {
		return tanh(input);
	}
	std::vector<double> Softmax(std::vector<double> input) {
		std::vector<double> output;
		double sum = 0;
		for (int i = 0; i < input.size(); i++) {
			output[i] = exp(input[i]);
			sum += output[i];
		}
		for (int i = 0; i < input.size(); i++) {
			output[i] /= sum;
		}
		return output;
	}
}

Layer::Layer(int numOfNodes, int numOfIncomingNodes, double (*activationFunction)(double))
	: numOfNodes(numOfNodes), numOfIncomingNodes(numOfIncomingNodes), activationFunction(activationFunction) {
	weights = std::vector<std::vector<double>>(numOfNodes, std::vector<double>(numOfIncomingNodes, 0));
	weightCostGradients = weights;
	biases = std::vector<double>(numOfNodes, 1);
	biasCostGradients = biases;
	srand(time(0));
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weights[i][j] = ((double)(rand() % 10000) - 5000.0) / 10000.0;
		}
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

		if (activationFunction != nullptr)
			output[i] = activationFunction(output[i]);
	}

	return output;
}

NeuralNetwork::NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double)) {
	for (int i = 0; i < numberOfNeurons.size()-1; i++) {
		layers.push_back(Layer(numberOfNeurons[i+1], numberOfNeurons[i], hiddenLayerAF));
	}
	//layers[numberOfNeurons.size() - 2].SetActivationFunction(outputLayerAF);
}

std::vector<double> NeuralNetwork::CalculateOutput(std::vector<double> input) {
	for (auto& layer : layers) {
		input = layer.FeedForward(input);
	}
	return input;
}

double NeuralNetwork::Cost(std::vector<double> actualOutput, std::vector<double> expectedOutput) {
	double cost = 0.0f;
	int size = actualOutput.size();

	for (int i = 0; i < size; i++) {
		cost += pow(actualOutput[i] - expectedOutput[i], 2);
	}
	cost /= size;

	return cost;
}

void NeuralNetwork::Learn(std::vector<double> trainingInputData, std::vector<double> expectedOutput, double learningRate) {
	float h = 0.0001f;
	double originalCost = Cost(CalculateOutput(trainingInputData), expectedOutput);

	for (auto& layer : layers) {
		for (int i = 0; i < layer.numOfNodes; i++) {
			for (int j = 0; j < layer.numOfIncomingNodes; j++) {
				layer.weights[i][j] += h;
				double newWeightCost = Cost(CalculateOutput(trainingInputData), expectedOutput);
				layer.weights[i][j] -= h;
				layer.weightCostGradients[i][j] = (newWeightCost - originalCost) / h;
			}

			layer.biases[i] += h;
			double newBiasCost = Cost(CalculateOutput(trainingInputData), expectedOutput);
			layer.biases[i] -= h;
			layer.biasCostGradients[i] = (newBiasCost - originalCost) / h;
		}
	}

	for (auto& layer : layers) {
		for (int i = 0; i < layer.numOfNodes; i++) {
			for (int j = 0; j < layer.numOfIncomingNodes; j++) {
				layer.weights[i][j] -= layer.weightCostGradients[i][j] * learningRate;
			}
			layer.biases[i] -= layer.biasCostGradients[i] * learningRate;
		}
	}
}