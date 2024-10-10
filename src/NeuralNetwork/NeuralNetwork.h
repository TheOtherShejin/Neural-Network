#pragma once

#include <iostream>
#include <math.h>
#include <vector>
#include "Layer.h"

struct DataPoint {
	Eigen::VectorXd input, expectedOutput;
	DataPoint(Eigen::VectorXd input, Eigen::VectorXd expectedOutput)
		: input(input), expectedOutput(expectedOutput) {}
};

class NeuralNetwork {
private:
	void ApplyAllGradients(double learningRate, int miniBatchSize);
	void ClearAllGradients();
	int inputSize;
public:
	std::vector<Layer> layers;
	
	NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
	void RandomizeAllParameters();
	Eigen::VectorXd Evaluate(Eigen::VectorXd input);
	double Cost(Eigen::VectorXd actualOutput, Eigen::VectorXd expectedOutput);
	void Learn(DataPoint dataPoint, double learningRate);
	void Learn(std::vector<DataPoint> dataset, double learningRate, int miniBatchSize);
	int GetInputSize() const;
	void SetActivationFunctions(double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
};