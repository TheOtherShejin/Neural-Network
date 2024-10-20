#pragma once

#include <iostream>
#include <math.h>
#include <vector>
#include "Layer.h"
#include <chrono>
#include <algorithm>
#include <random>
#include "Vector.h"
#include "Matrix.h"

struct DataPoint {
	Vector input, expectedOutput;

	DataPoint(Vector input, Vector expectedOutput)
		: input(input), expectedOutput(expectedOutput) {}
};

class NeuralNetwork {
private:
	void ApplyAllGradients(double learningRate, int miniBatchSize);
	void ClearAllGradients();

	void BackPropagate(DataPoint* dataPoint, Vector* actualOutput);
	int inputSize;
public:
	std::vector<Layer> layers;
	
	NeuralNetwork(std::vector<int> numberOfNeurons, double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
	void RandomizeAllParameters();

	Vector Evaluate(Vector input);
	void Learn(std::vector<DataPoint> dataset, double learningRate, int miniBatchSize);
	void SGD(std::vector<DataPoint>* dataset, int epochs, double learningRate, int miniBatchSize);
	double Cost(Vector actualOutput, Vector expectedOutput);

	int GetInputSize() const;
	void SetActivationFunctions(double (*hiddenLayerAF)(double), double (*outputLayerAF)(double));
};