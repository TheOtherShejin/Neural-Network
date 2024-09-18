#include <iostream>
#include <chrono>
#include "NeuralNetwork.h"

double learningRate = 1.5f;
int epochs = 10000;

int main() {
	NeuralNetwork nn({ 2, 4, 1 }, ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid);

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
	}
	auto endTime = std::chrono::high_resolution_clock::now();
	auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	std::cout << "Elapsed Time For Training: " << dt.count()*0.001f << "s\n\n";

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
	std::cout << "Average Loss: " << avgLoss << '\n';

	return 0;
}