#pragma once

#include <iostream>
#include <math.h>
#include <vector>
#include "Layer.h"

struct DataPoint {
	std::vector<double> input;
	std::vector<double> expectedOutput;
	DataPoint(std::vector<double> input, std::vector<double> expectedOutput)
		: input(input), expectedOutput(expectedOutput) {}
};

class NeuralNetwork {
private:
	void ApplyAllGradients(double learningRate);
	void ClearAllGradients();
	int inputSize;
public:
	std::vector<Layer> layers;
	
	NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
	std::vector<double> CalculateOutput(std::vector<double> input);
	double Loss(std::vector<double> actualOutput, std::vector<double> expectedOutput);
	void Learn(DataPoint dataPoint, double learningRate);
	void Learn(std::vector<DataPoint> dataPoints, double learningRate);
	int GetInputSize() const;
	void SetActivationFunctions(double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
};