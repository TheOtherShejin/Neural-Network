#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double)) {
	inputSize = numberOfNeurons[0];
	for (int i = 1; i < numberOfNeurons.size(); i++) {
		layers.push_back(Layer(numberOfNeurons[i], numberOfNeurons[i-1], hiddenLayerAF));
	}
	layers[layers.size()-1].activationFunction = outputLayerAF;
}

void NeuralNetwork::RandomizeAllParameters() {
	for (auto& layer : layers) {
		layer.RandomizeParameters();
	}
}

Eigen::VectorXd NeuralNetwork::Evaluate(Eigen::VectorXd input) {
	for (int i = 0; i < layers.size(); i++) {
		input = layers[i].FeedForward(input);
	}
	return input;
}

double NeuralNetwork::Cost(Eigen::VectorXd actualOutput, Eigen::VectorXd expectedOutput) {
	return (actualOutput - expectedOutput).squaredNorm() * 0.5;
}

void NeuralNetwork::Learn(DataPoint dataPoint, double learningRate) {
	Layer& outputLayer = layers[layers.size() - 1];
	Eigen::VectorXd actualOutput = Evaluate(dataPoint.input);

	Eigen::VectorXd errors = outputLayer.CalculateOutputLayerErrors(actualOutput, dataPoint.expectedOutput);
	outputLayer.UpdateGradients(errors);

	for (int i = layers.size() - 2; i >= 0; i--) {
		errors = layers[i].CalculateHiddenLayerErrors(layers[i + 1], errors);
		layers[i].UpdateGradients(errors);
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

int NeuralNetwork::GetInputSize() const {
	return inputSize;
}

void NeuralNetwork::SetActivationFunctions(double (*hiddenLayerAF)(double), double (*outputLayerAF)(double)) {
	for (auto& layer : layers) {
		layer.activationFunction = hiddenLayerAF;
	}
	layers[layers.size() - 1].activationFunction = outputLayerAF;
}