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

typedef std::vector<DataPoint> Dataset;

class NeuralNetwork {
private:
	void ApplyAllGradients(double learningRate, int miniBatchSize);
	void ClearAllGradients();

	void BackPropagate(DataPoint* dataPoint, Vector* actualOutput);
	int inputSize;

	Vector (*CostFuncDerivative)(Vector, Vector) = nullptr;
public:
	std::vector<Layer> layers;
	double (*CostFunction)(Vector, Vector) = nullptr;
	
	NeuralNetwork(std::vector<int> numberOfNeurons, Vector (*hiddenLayerAF)(Vector) = AF::Sigmoid, Vector (*outputLayerAF)(Vector) = AF::Sigmoid, double (*CostFunction)(Vector, Vector) = Cost::MeanSquaredError);
	void RandomizeAllParameters();

	Vector Evaluate(Vector input);
	void Learn(std::vector<DataPoint> dataset, double learningRate, int miniBatchSize);
	void SGD(std::vector<DataPoint>* dataset, int epochs, double learningRate, int miniBatchSize, std::vector<DataPoint>* validation_dataset = nullptr);
	double Cost(Vector actualOutput, Vector expectedOutput);

	int GetInputSize() const;
	void SetActivationFunctions(Vector (*hiddenLayerAF)(Vector), Vector (*outputLayerAF)(Vector));
};