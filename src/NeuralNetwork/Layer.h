#pragma once

#include <time.h>
#include <math.h>
#include <vector>
#include "ActivationFunction.h"
#include <iostream>
#include <Eigen/Eigen>

class Layer {
public:
	double (*activationFunction)(double) = nullptr;
	Eigen::VectorXd weightedInputs;
	Eigen::VectorXd rawInputs;

	int numOfNodes, numOfIncomingNodes;

	Eigen::MatrixXd weights;
	Eigen::MatrixXd weightCostGradients;
	Eigen::VectorXd biases;
	Eigen::VectorXd biasCostGradients;

	Layer(int numOfNodes, int numOfIncomingNodes, double (*activationFunction)(double));
	void RandomizeParameters();
	Eigen::VectorXd FeedForward(Eigen::VectorXd input);

	void ApplyGradients(double learningRate);
	void UpdateGradients(Eigen::VectorXd nodeValues);
	void ClearGradients();

	Eigen::VectorXd CostDerivative(Eigen::VectorXd actualOutput, Eigen::VectorXd expectedOutput);
	Eigen::VectorXd CalculateOutputLayerErrors(Eigen::VectorXd actualOutput, Eigen::VectorXd expectedOutput);
	Eigen::VectorXd CalculateHiddenLayerErrors(Layer& nextLayer, Eigen::VectorXd nextLayerErrors);
};

Eigen::VectorXd Sigmoid(Eigen::VectorXd input);
Eigen::VectorXd SigmoidDerivative(Eigen::VectorXd input);