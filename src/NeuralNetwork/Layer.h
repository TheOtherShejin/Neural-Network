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
	double (*activationFunction)(double) = nullptr;
	Vector weightedInputs;
	Vector rawInputs;

	int numOfNodes, numOfIncomingNodes;

	Matrix weights;
	Matrix weightCostGradients;
	Vector biases;
	Vector biasCostGradients;

	Layer(int numOfNodes, int numOfIncomingNodes, double (*activationFunction)(double));
	void RandomizeParameters();
	Vector FeedForward(Vector input);

	void ApplyGradients(double learningRate, int miniBatchSize);
	void UpdateGradients(Vector nodeValues);
	void ClearGradients();

	Vector CostDerivative(Vector actualOutput, Vector expectedOutput);
	Vector CalculateOutputLayerErrors(Vector actualOutput, Vector expectedOutput);
	Vector CalculateHiddenLayerErrors(Layer& nextLayer, Vector nextLayerErrors);
};

Vector Sigmoid(Vector input);
double Sigmoid(double input);
Vector SigmoidDerivative(Vector input);
double SigmoidDerivative(double input);