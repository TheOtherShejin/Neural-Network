#pragma once

#include <time.h>
#include <math.h>
#include <vector>
#include "ActivationFunction.h"
#include "Vector.h"
#include "Matrix.h"
#include <iostream>

class Layer {
public:
	Vector (*ActivationFunction)(Vector) = nullptr;
	Vector (*ActivationDerivative)(Vector) = nullptr;
	Vector weightedInputs;
	Vector rawInputs;

	int numOfNodes, numOfIncomingNodes;

	Matrix weights;
	Matrix weightCostGradients;
	Vector biases;
	Vector biasCostGradients;

	Layer(int numOfNodes, int numOfIncomingNodes, Vector (*ActivationFunction)(Vector));
	void RandomizeParameters();
	Vector FeedForward(Vector input);
	void SetActivationFunction(Vector (*ActivationFunction)(Vector));

	void ApplyGradients(double learningRate, int miniBatchSize);
	void UpdateGradients(Vector nodeValues);
	void ClearGradients();

	Vector CostDerivative(Vector actualOutput, Vector expectedOutput);
	Vector CalculateOutputLayerErrors(Vector actualOutput, Vector expectedOutput);
	Vector CalculateHiddenLayerErrors(Layer& nextLayer, Vector nextLayerErrors);
};