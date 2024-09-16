#include <iostream>
#include "NeuralNetwork.h"

int main() {
	NeuralNetwork nn({ 2, 3, 1 }, ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid);
	for (int i = 0; i < 1000; i++) {
		nn.Learn({ 0, 0 }, { 0 }, 1.5f);
		nn.Learn({ 0, 1 }, { 1 }, 1.5f);
		nn.Learn({ 1, 0 }, { 1 }, 1.5f);
		nn.Learn({ 1, 1 }, { 0 }, 1.5f);
	}

	std::vector<double> output;
	for (int i = 0; i < 4; i++) {
		int a = i & 1;
		int b = (i & 2) >> 1;
		output = nn.CalculateOutput({ (double)a, (double)b });
		std::cout << (output[0] > 0.5f ? 1 : 0) << ' ' << output[0] << '\n';
	}

	return 0;
}