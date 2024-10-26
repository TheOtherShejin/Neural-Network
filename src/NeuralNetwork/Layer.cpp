#include "Layer.h"

Layer::Layer(int numOfNodes, int numOfIncomingNodes, double (*activationFunction)(double))
	: numOfNodes(numOfNodes), numOfIncomingNodes(numOfIncomingNodes), activationFunction(activationFunction) {

	weights = Matrix(numOfNodes, numOfIncomingNodes);
	weightCostGradients = weights;
	biases = Vector(numOfNodes);
	biasCostGradients = biases;
	RandomizeParameters();
}

void Layer::RandomizeParameters() {
	srand(time(0));
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weights(i, j) = ((double)(rand() % 10000) - 5000.0) / 10000.0;
		}
		biases(i) = ((double)(rand() % 10000) - 5000.0) / 10000.0;
	}
}

Vector Layer::FeedForward(Vector input) {
	/*Vector output = weights * input + biases;
	weightedInputs = output;
	rawInputs = input;
	output = Sigmoid(output);
	return output;*/

	Vector output(numOfNodes);
	rawInputs = input;
	weightedInputs = input;
	for (int i = 0; i < numOfNodes; i++) {
		weightedInputs(i) = biases(i);
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weightedInputs(i) += input(j) * weights(i, j);
		}
		output(i) = Sigmoid(weightedInputs(i));
	}

	return output;
}

void Layer::ApplyGradients(double learningRate, int miniBatchSize) {
	weights -= weightCostGradients * (learningRate / (double)miniBatchSize);
	biases -= biasCostGradients * (learningRate / (double)miniBatchSize);
}

void Layer::UpdateGradients(Vector errors) {
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weightCostGradients(i, j) += rawInputs(j) * errors(i);
		}
		biasCostGradients(i) += errors(i);
	}
	//weightCostGradients += errors * rawInputs.transpose();
	//biasCostGradients += errors;
}

void Layer::ClearGradients() {
	weightCostGradients.SetZero();
	biasCostGradients.SetZero();
}

Vector Layer::CalculateOutputLayerErrors(Vector actualOutput, Vector expectedOutput) {
	return CostDerivative(actualOutput, expectedOutput) * SigmoidDerivative(weightedInputs);

	/*Vector errors = CostDerivative(actualOutput, expectedOutput);
	for (int i = 0; i < errors.size; i++) {
		errors(i) *= SigmoidDerivative(weightedInputs(i));
	}
	return errors;*/
}

Vector Layer::CalculateHiddenLayerErrors(Layer& nextLayer, Vector nextLayerErrors) {
	/*Vector errors = nextLayer.weights.transpose() * nextLayerErrors;
	errors = errors.cwiseProduct(SigmoidDerivative(weightedInputs));
	return errors;*/
	
	Vector errors(numOfNodes);
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < nextLayer.numOfNodes; j++) {
			errors(i) += nextLayerErrors(j) * nextLayer.weights(j, i);
		}
		errors(i) *= SigmoidDerivative(weightedInputs(i));
	}
	return errors;
}

Vector Layer::CostDerivative(Vector actualOutput, Vector expectedOutput) {
	return actualOutput - expectedOutput;
}

Vector Sigmoid(Vector input) {
	Vector output = input;
	for (int i = 0; i < input.size; i++) {
		output(i) = 1.0f / (1.0f + exp(-input(i)));
	}
	return output;
}

Vector SigmoidDerivative(Vector input) {
	Vector value = Sigmoid(input);
	Vector output(input.size);

	for (int i = 0; i < output.size; i++) {
		output(i) = value(i) * (1 - value(i));
	}

	return output;
}

double Sigmoid(double input) {
	return 1.0f / (1.0f + exp(-input));
}

double SigmoidDerivative(double input) {
	double value = Sigmoid(input);
	return value * (1 - value);
}