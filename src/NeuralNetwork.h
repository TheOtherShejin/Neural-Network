#pragma once

#include <iostream>
#include <math.h>
#include <vector>

namespace ActivationFunctions {
	double ReLU(double input);
	double Sigmoid(double input);
	double HyperbolicTangent(double input);
}

class Layer {
	double (*activationFunction)(double) = nullptr;
public:
	int numOfNodes, numOfIncomingNodes;
	std::vector<std::vector<double>> weights;
	std::vector<double> biases;

	Layer(int numOfNodes, int numOfIncomingNodes, double (*activationFunction)(double));
	void SetActivationFunction(double (*activationFunction)(double));
	std::vector<double> FeedForward(std::vector<double>& input);
};

class NeuralNetwork {
public:
	std::vector<Layer> layers;
	
	NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
	std::vector<double> CalculateOutput(std::vector<double> input);
};