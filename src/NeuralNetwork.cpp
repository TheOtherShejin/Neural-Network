#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double)) {
	for (int i = 1; i < numberOfNeurons.size(); i++) {
		layers.push_back(Layer(numberOfNeurons[i], numberOfNeurons[i-1], hiddenLayerAF));
	}
	layers[layers.size()-1].activationFunction = outputLayerAF;
}

std::vector<double> NeuralNetwork::CalculateOutput(std::vector<double> input) {
	for (int i = 0; i < layers.size(); i++) {
		input = layers[i].FeedForward(input);
	}
	return input;
}

double NeuralNetwork::Loss(std::vector<double> actualOutput, std::vector<double> expectedOutput) {
	double cost = 0.0f;
	int size = actualOutput.size();

	for (int i = 0; i < size; i++) {
		cost += pow(actualOutput[i] - expectedOutput[i], 2);
	}
	cost /= size;

	return cost;
}

void NeuralNetwork::Learn(DataPoint dataPoint, double learningRate) {
	Layer& outputLayer = layers[layers.size() - 1];
	std::vector<double> actualOutput = CalculateOutput(dataPoint.input);
	std::vector<double> nodeValues = outputLayer.CalculateOutputLayerNodeValues(actualOutput, dataPoint.expectedOutput);
	outputLayer.UpdateGradients(nodeValues);


	for (int i = layers.size() - 2; i >= 0; i--) {
		nodeValues = layers[i].CalculateHiddenLayerNodeValues(layers[i + 1], nodeValues);
		layers[i].UpdateGradients(nodeValues);
	}

	ApplyAllGradients(learningRate);
	ClearAllGradients();
}

void NeuralNetwork::Learn(std::vector<DataPoint> dataPoints, double learningRate) {
	for (auto& dataPoint : dataPoints) {
		Learn(dataPoint, learningRate);
	}
}

void NeuralNetwork::ApplyAllGradients(double learningRate) {
	for (auto& layer : layers) {
		layer.ApplyGradients(learningRate);
	}
}

void NeuralNetwork::ClearAllGradients() {
	for (auto& layer : layers) {
		layer.ClearGradients();
	}
}