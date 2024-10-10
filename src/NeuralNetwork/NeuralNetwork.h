#pragma once

#include <iostream>
#include <math.h>
#include <vector>
#include "Layer.h"
#include <chrono>
#include <algorithm>
#include <random>

struct DataPoint {
	Eigen::VectorXd input, expectedOutput;
	DataPoint(Eigen::VectorXd input, Eigen::VectorXd expectedOutput)
		: input(input), expectedOutput(expectedOutput) {}
};

class NeuralNetwork {
private:
	void ApplyAllGradients(double learningRate, int miniBatchSize);
	void ClearAllGradients();

	void BackPropagate(DataPoint* dataPoint, Eigen::VectorXd* actualOutput);
	int inputSize;
public:
	std::vector<Layer> layers;
	
	NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
	void RandomizeAllParameters();

	Eigen::VectorXd Evaluate(Eigen::VectorXd input);
	void Learn(std::vector<DataPoint> dataset, double learningRate, int miniBatchSize);
	void SGD(std::vector<DataPoint>* dataset, int epochs, double learningRate, int miniBatchSize);
	double Cost(Eigen::VectorXd actualOutput, Eigen::VectorXd expectedOutput);

	int GetInputSize() const;
	void SetActivationFunctions(double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
};