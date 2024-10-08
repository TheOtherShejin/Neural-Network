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

void NeuralNetwork::BackPropagate(DataPoint* dataPoint, Eigen::VectorXd* actualOutput) {
	Layer& outputLayer = layers[layers.size() - 1];

	// Error
	Eigen::VectorXd errors = outputLayer.CalculateOutputLayerErrors(*actualOutput, dataPoint->expectedOutput);
	outputLayer.UpdateGradients(errors);

	for (int i = layers.size() - 2; i >= 0; i--) {
		errors = layers[i].CalculateHiddenLayerErrors(layers[i + 1], errors);
		layers[i].UpdateGradients(errors);
	}
}

void NeuralNetwork::SGD(std::vector<DataPoint>* dataset, int epochs, double learningRate, int miniBatchSize) {
	float totalTimeTaken = 0.0f;
	std::cout << "Training Started - " << epochs << " Epochs, Learning Rate: " << learningRate << ", Mini-Batch Size: " << miniBatchSize << '\n';
	for (int i = 0; i < epochs; i++) {
		std::cout << "Epoch: " << i;

		// Randomize Train
		std::shuffle(dataset->begin(), dataset->end(), std::mt19937{std::random_device{}()});

		auto startTime = std::chrono::high_resolution_clock::now();
		Learn(*dataset, learningRate, miniBatchSize);
		auto endTime = std::chrono::high_resolution_clock::now();
		auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

		std::cout << " - " << (dt.count() * 0.001f) << "s\n";
		totalTimeTaken += dt.count() * 0.001f;
	}
	std::cout << "Training Completed in " << totalTimeTaken << "s\n";
}

void NeuralNetwork::Learn(std::vector<DataPoint> dataset, double learningRate, int miniBatchSize) {
	for (int i = 0; i < (dataset.size() / miniBatchSize); i++) {
		for (int j = 0; j < miniBatchSize; j++) {
			int index = j + i * miniBatchSize;
			// Feedforward
			Eigen::VectorXd actualOutput = Evaluate(dataset[index].input);

			BackPropagate(&dataset[index], &actualOutput);
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