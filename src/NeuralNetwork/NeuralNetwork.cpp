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
	// Feedforward
	Eigen::VectorXd actualOutput = Evaluate(dataPoint.input);
	Layer& outputLayer = layers[layers.size() - 1];

	// Error
	Eigen::VectorXd errors = outputLayer.CalculateOutputLayerErrors(actualOutput, dataPoint.expectedOutput);
	outputLayer.UpdateGradients(errors);

	for (int i = layers.size() - 2; i >= 0; i--) {
		errors = layers[i].CalculateHiddenLayerErrors(layers[i + 1], errors);
		layers[i].UpdateGradients(errors);
	}

	// Gradient Descent
	ApplyAllGradients(learningRate, 1);
	ClearAllGradients();
}

void NeuralNetwork::Learn(std::vector<DataPoint> dataset, double learningRate, int miniBatchSize) {
	/*for (auto& dataPoint : dataPoints) {
		Learn(dataPoint, learningRate);
	}*/

	for (int i = 0; i < (dataset.size() / miniBatchSize); i++) {
		for (int j = 0; j < miniBatchSize; j++) {
			int index = j + i * miniBatchSize;
			// Feedforward
			Eigen::VectorXd actualOutput = Evaluate(dataset[index].input);
			Layer& outputLayer = layers[layers.size() - 1];

			// Error
			Eigen::VectorXd errors = outputLayer.CalculateOutputLayerErrors(actualOutput, dataset[index].expectedOutput);
			outputLayer.UpdateGradients(errors);

			for (int i = layers.size() - 2; i >= 0; i--) {
				errors = layers[i].CalculateHiddenLayerErrors(layers[i + 1], errors);
				layers[i].UpdateGradients(errors);
			}
		}

		// Gradient Descent
		ApplyAllGradients(learningRate, miniBatchSize);
		ClearAllGradients();
	}
}

void NeuralNetwork::ApplyAllGradients(double learningRate, int miniBatchSize) {
	for (auto& layer : layers) {
		layer.ApplyGradients(learningRate, miniBatchSize);
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