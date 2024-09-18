#pragma once

#include <iostream>
#include <math.h>
#include <vector>
#include "Layer.h"

struct DataPoint {
	std::vector<double> input;
	std::vector<double> expectedOutput;
};

class NeuralNetwork {
private:
	void ApplyAllGradients(double learningRate);
	void ClearAllGradients();
public:
	std::vector<Layer> layers;
	
	NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
	std::vector<double> CalculateOutput(std::vector<double> input);
	double Loss(std::vector<double> actualOutput, std::vector<double> expectedOutput);
	void Learn(DataPoint dataPoint, double learningRate);
	void Learn(std::vector<DataPoint> dataPoints, double learningRate);
};