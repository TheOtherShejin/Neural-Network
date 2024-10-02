#include <iostream>
#include <chrono>
#include "NeuralNetwork.h"
#include "ModelExporter.h"

double learningRate = 1.5f;
int epochs = 10000;

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