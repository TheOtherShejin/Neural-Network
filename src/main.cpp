#include <iostream>
#include <chrono>
#include "NeuralNetwork.h"
#include <string>
#include <fstream>
#include <sstream>

double learningRate = 1.5f;
int epochs = 10000;

void SaveModelToCSV(std::string path, NeuralNetwork* nn) {
	std::ofstream file;
	file.open(path);

	// Write Number of Layers
	file << nn->GetInputSize() << ',';
	for (int i = 0; i < nn->layers.size(); i++) {
		file << nn->layers[i].numOfNodes << ',';
	}
	file << '\n';

	// Write Activation Functions
	file << AF::GetFunctionEnum(nn->layers[0].activationFunction) << ',';
	file << AF::GetFunctionEnum(nn->layers[nn->layers.size()-1].activationFunction) << '\n';

	// Write Weights of Each Layer
	for (auto& layer : nn->layers) {
		for (int i = 0; i < layer.numOfNodes; i++) {
			for (int j = 0; j < layer.numOfIncomingNodes; j++) {
				file << layer.weights[i][j] << ',';
			}
		}
		file << '\n';
	}

	// Write Biases of Each Layer
	for (auto& layer : nn->layers) {
		for (int i = 0; i < layer.numOfNodes; i++) {
			file << layer.biases[i] << ',';
		}
		file << '\n';
	}

	file.close();
	std::cout << "Model successfully saved at " << path << '\n';
}

NeuralNetwork LoadModelFromCSV(std::string path) {
	std::ifstream file;
	file.open(path);

	std::vector<int> numberOfNeurons;
	double (*hiddenLayerAF)(double) = AF::Linear, (*outputLayerAF)(double) = AF::Linear;

	// Load Number Of Layers
	if (file.good()) {
		std::string line;
		std::string substring;
		std::getline(file, line);
		std::stringstream ss(line);
		while (std::getline(ss, substring, ',')) {
			numberOfNeurons.push_back(std::stoi(substring));
		}

		std::getline(file, line);
		ss = std::stringstream(line);
		std::getline(ss, substring, ',');
		hiddenLayerAF = AF::GetFunctionFromEnum(AF::FunctionType(std::stoi(substring)));
		std::getline(ss, substring, ',');
		outputLayerAF = AF::GetFunctionFromEnum(AF::FunctionType(std::stoi(substring)));
	}

	NeuralNetwork nn(numberOfNeurons, hiddenLayerAF, outputLayerAF);

	// Load Weights Of Each Layer
	for (int i = 0; i < numberOfNeurons.size() - 1; i++) {
		std::string line;
		std::string substring;
		std::vector<double> weights;
		std::getline(file, line);
		std::stringstream ss(line);
		while (std::getline(ss, substring, ',')) {
			weights.push_back(std::stod(substring));
		}

		Layer& layer = nn.layers[i];
		for (int j = 0; j < layer.numOfNodes; j++) {
			for (int k = 0; k < layer.numOfIncomingNodes; k++) {
				layer.weights[j][k] = weights[j * layer.numOfIncomingNodes + k];
			}
		}
	}

	// Load Biases Of Each Layer
	for (int i = 0; i < numberOfNeurons.size() - 1; i++) {
		std::string line;
		std::string substring;
		std::vector<double> biases;
		std::getline(file, line);
		std::stringstream ss(line);
		while (std::getline(ss, substring, ',')) {
			biases.push_back(std::stod(substring));
		}

		Layer& layer = nn.layers[i];
		for (int j = 0; j < layer.numOfNodes; j++) {
			layer.biases[j] = biases[j];
		}
	}

	return nn;
}

int main() {
	char startDecision;
	std::cout << "Train a new model, or Load a model from csv? (T/L): ";
	std::cin >> startDecision;
	
	NeuralNetwork nn({ 2, 3, 1 }, ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid);
	if (std::tolower(startDecision) == 't') {
		std::vector<DataPoint> dataPoints = {
			{ {0, 0}, {0} },
			{ {0, 1}, {1} },
			{ {1, 0}, {1} },
			{ {1, 1}, {0} }
		};

		// Training
		std::cout << "Training Started - " << epochs << " Epochs, Learning Rate: " << learningRate << '\n';
		auto startTime = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < epochs; i++) {
			nn.Learn(dataPoints, learningRate);

			if (i % 1000 == 0) {
				std::vector<double> output;
				double avgLoss = 0.0f;
				for (int j = 0; j < 4; j++) {
					int a = j & 1;
					int b = (j & 2) >> 1;
					output = nn.CalculateOutput({ (double)a, (double)b });
					avgLoss += nn.Loss(output, { (double)(a ^ b) });
				}
				avgLoss /= 4;
				std::cout << "Epoch: " << i << ", Average Loss: " << avgLoss << '\n';
			}
		}
		auto endTime = std::chrono::high_resolution_clock::now();
		auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
		std::cout << "Elapsed Time For Training: " << dt.count() * 0.001f << "s\n\n";
	}
	else if (std::tolower(startDecision) == 'l') {
		std::string path;
		std::cout << "Enter .csv file path: ";
		std::cin >> path;
		nn = LoadModelFromCSV(path);
	}
	else return 0;

	char shouldTest;
	std::cout << "Test Model? (Y/N): ";
	std::cin >> shouldTest;

	if (std::tolower(shouldTest) == 'y') {
		// Output
		std::vector<double> output;
		double avgLoss = 0.0f;
		std::cout << "XOR Problem Example:\n";
		for (int i = 0; i < 4; i++) {
			int a = i & 1;
			int b = (i & 2) >> 1;
			output = nn.CalculateOutput({ (double)a, (double)b });
			avgLoss += nn.Loss(output, { (double)(a ^ b) });
			std::cout << (output[0] > 0.5f ? 1 : 0) << ' ' << output[0] << '\n';
		}
		avgLoss /= 4;
		std::cout << "Average Loss: " << avgLoss << "\n\n";
	}

	char shouldSave;
	std::cout << "Save Model? (Y/N): ";
	std::cin >> shouldSave;
	if (std::tolower(shouldSave) == 'y') {
		std::string path;
		std::cout << "Enter save location: ";
		std::cin >> path;
		SaveModelToCSV(path, &nn);
	}

	return 0;
}