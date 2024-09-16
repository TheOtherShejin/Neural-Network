#include <iostream>
#include "NeuralNetwork.h"

int main() {
	NeuralNetwork nn({ 2, 2, 1 }, ActivationFunctions::ReLU, ActivationFunctions::Sigmoid);

	std::vector<double> output = nn.CalculateOutput({ 0.0f, 1.0f });
	std::cout << output[0] << '\n';

	return 0;
}