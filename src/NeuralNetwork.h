#pragma once

#include <iostream>
#include <math.h>
#include <vector>
#include "Layer.h"

class NeuralNetwork {
public:
	std::vector<Layer> layers;
	
	NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
	std::vector<double> CalculateOutput(std::vector<double> input);
	double Cost(std::vector<double> actualOutput, std::vector<double> expectedOutput);
	void Learn(std::vector<double> trainingInputData, std::vector<double> expectedOutput, double learningRate);
};