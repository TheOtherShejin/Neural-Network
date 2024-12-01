#pragma once

#include <time.h>
#include <math.h>
#include <vector>
#include <NeuralNetwork/ActivationFunction.h>
#include <NeuralNetwork/Cost.h>
#include <NeuralNetwork/Maths/Vector.h>
#include <NeuralNetwork/Maths/Matrix.h>
#include <iostream>

class Layer {
public:
	AF::ActivationFunction* activationFunction;

	Vector weightedInputs;
	Vector rawInputs;

	int numOfNodes, numOfIncomingNodes;

	Matrix weights;
	Matrix weightCostGradients;
	Vector biases;
	Vector biasCostGradients;

	Layer(int numOfNodes, int numOfIncomingNodes, AF::FunctionType activationFunctionType);
	void RandomizeParameters();
	Vector FeedForward(Vector input);
	void SetActivationFunction(AF::FunctionType activationFunctionType);

	void ApplyGradients(double learningRate, int miniBatchSize, double lambda, int datasetSize);
	void UpdateGradients(Vector nodeValues);
	void ClearGradients();

	Vector CalculateOutputLayerErrors(Vector actualOutput, Vector expectedOutput, Cost::CostFunction* costFunction);
	Vector CalculateHiddenLayerErrors(Layer& nextLayer, Vector nextLayerErrors);
};