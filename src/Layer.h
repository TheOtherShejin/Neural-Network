#pragma once

#include <time.h>
#include <math.h>
#include <vector>
#include "ActivationFunction.h"
#include <iostream>

class Layer {
public:
	double (*activationFunction)(double) = nullptr;
	std::vector<double> weightedInputs;
	std::vector<double> rawInputs;

	int numOfNodes, numOfIncomingNodes;
	std::vector<std::vector<double>> weights;
	std::vector<std::vector<double>> weightCostGradients;
	std::vector<double> biases;
	std::vector<double> biasCostGradients;

	Layer(int numOfNodes, int numOfIncomingNodes, double (*activationFunction)(double));
	std::vector<double> FeedForward(std::vector<double> input);

	void ApplyGradients(double learningRate);
	void UpdateGradients(std::vector<double> nodeValues);
	void ClearGradients();

	double LossDerivative(double actualOutput, double expectedOutput);
	std::vector<double> CalculateOutputLayerNodeValues(std::vector<double> actualOutput, std::vector<double> expectedOutput);
	std::vector<double> CalculateHiddenLayerNodeValues(Layer& oldLayer, std::vector<double> oldNodeValues);
};