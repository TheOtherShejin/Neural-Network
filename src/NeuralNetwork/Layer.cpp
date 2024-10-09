#include "Layer.h"

Layer::Layer(int numOfNodes, int numOfIncomingNodes, double (*activationFunction)(double))
	: numOfNodes(numOfNodes), numOfIncomingNodes(numOfIncomingNodes), activationFunction(activationFunction) {

	weights = Eigen::MatrixXd(numOfNodes, numOfIncomingNodes);
	weightCostGradients = Eigen::MatrixXd::Zero(numOfNodes, numOfIncomingNodes);
	biases = Eigen::VectorXd(numOfNodes);
	biasCostGradients = Eigen::VectorXd::Zero(numOfNodes);

	srand(time(0));
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weights(i, j) = ((double)(rand() % 10000) - 5000.0) / 10000.0;
		}
		biases(i) = ((double)(rand() % 10000) - 5000.0) / 10000.0;
	}
}

Eigen::VectorXd Layer::FeedForward(Eigen::VectorXd input) {
	Eigen::VectorXd output = weights * input + biases;
	weightedInputs = output;
	rawInputs = input;
	output = Sigmoid(output);

	return output;
}

void Layer::ApplyGradients(double learningRate) {
	weights -= weightCostGradients * learningRate;
	biases -= biasCostGradients * learningRate;
}

void Layer::UpdateGradients(Eigen::VectorXd errors) {
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weightCostGradients(i, j) = rawInputs(j) * errors(i);
		}
	}
	//weightCostGradients += errors * rawInputs.transpose();
	biasCostGradients = errors;
}

void Layer::ClearGradients() {
	weightCostGradients.setZero();
	biasCostGradients.setZero();
}

Eigen::VectorXd Layer::CalculateOutputLayerErrors(Eigen::VectorXd actualOutput, Eigen::VectorXd expectedOutput) {
	Eigen::VectorXd errors = CostDerivative(actualOutput, expectedOutput);
	errors = errors.cwiseProduct(SigmoidDerivative(weightedInputs));
	return errors;
}

Eigen::VectorXd Layer::CalculateHiddenLayerErrors(Layer& nextLayer, Eigen::VectorXd nextLayerErrors) {
	Eigen::VectorXd errors = nextLayer.weights.transpose() * nextLayerErrors;
	errors = errors.cwiseProduct(SigmoidDerivative(weightedInputs));

	return errors;
}

Eigen::VectorXd Layer::CostDerivative(Eigen::VectorXd actualOutput, Eigen::VectorXd expectedOutput) {
	return actualOutput - expectedOutput;
}

Eigen::VectorXd Sigmoid(Eigen::VectorXd input) {
	Eigen::VectorXd output = input;
	for (int i = 0; i < input.size(); i++) {
		output(i) = 1.0f / (1.0f + exp(-input(i)));
	}
	return output;
}

Eigen::VectorXd SigmoidDerivative(Eigen::VectorXd input) {
	Eigen::VectorXd value = Sigmoid(input);
	Eigen::VectorXd output(input.size());

	for (int i = 0; i < output.size(); i++) {
		output(i) = value(i) * (1 - value(i));
	}

	return output;
}