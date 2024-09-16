#pragma once

#include <iostream>
#include <math.h>
#include <vector>

namespace ActivationFunctions {
	double Linear(double input);
	double BinaryStep(double input);
	double ReLU(double input);
	double LeakyReLU(double input);
	double Sigmoid(double input);
	double TanH(double input);
	std::vector<double> Softmax(std::vector<double> input);
}

class Layer {
	double (*activationFunction)(double) = nullptr;
public:
	int numOfNodes, numOfIncomingNodes;
	std::vector<std::vector<double>> weights;
	std::vector<std::vector<double>> weightCostGradients;
	std::vector<double> biases;
	std::vector<double> biasCostGradients;

	Layer(int numOfNodes, int numOfIncomingNodes, double (*activationFunction)(double));
	void SetActivationFunction(double (*activationFunction)(double));
	std::vector<double> FeedForward(std::vector<double>& input);
};

class NeuralNetwork {
public:
	std::vector<Layer> layers;
	
	NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
	std::vector<double> CalculateOutput(std::vector<double> input);
	double Cost(std::vector<double> actualOutput, std::vector<double> expectedOutput);
	void Learn(std::vector<double> trainingInputData, std::vector<double> expectedOutput, double learningRate);
};