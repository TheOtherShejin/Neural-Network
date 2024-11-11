#include <NeuralNetwork/Layer.h>

Layer::Layer(int numOfNodes, int numOfIncomingNodes, Vector (*ActivationFunction)(Vector))
	: numOfNodes(numOfNodes), numOfIncomingNodes(numOfIncomingNodes), ActivationFunction(ActivationFunction) {

	ActivationDerivative = AF::GetDerivativeFromEnum(AF::GetFunctionEnum(ActivationFunction));

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
	Vector output = weights * input + biases;
	weightedInputs = output;
	rawInputs = input;
	output = ActivationFunction(output);
	return output;

	/*Vector output(numOfNodes);
	rawInputs = input;
	weightedInputs = Vector(numOfNodes);
	for (int i = 0; i < numOfNodes; i++) {
		weightedInputs(i) = biases(i);
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weightedInputs(i) += input(j) * weights(i, j);
		}
		output(i) = Sigmoid(weightedInputs(i));
	}
	return output;*/
}

void Layer::SetActivationFunction(Vector (*ActivationFunction)(Vector)) {
	this->ActivationFunction = ActivationFunction;
	this->ActivationDerivative = AF::GetDerivativeFromEnum(AF::GetFunctionEnum(ActivationFunction));
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

Vector Layer::CalculateOutputLayerErrors(Vector actualOutput, Vector expectedOutput, Vector(*CostDerivativeFunc)(Vector, Vector)) {
	return CostDerivative(actualOutput, expectedOutput, CostDerivativeFunc) * ActivationDerivative(weightedInputs);
}

Vector Layer::CalculateHiddenLayerErrors(Layer& nextLayer, Vector nextLayerErrors) {
	Vector errors = nextLayer.weights.Transpose() * nextLayerErrors;
	return errors * ActivationDerivative(weightedInputs);
	
	/*Vector errors(numOfNodes);
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < nextLayer.numOfNodes; j++) {
			errors(i) += nextLayerErrors(j) * nextLayer.weights(j, i);
		}
		errors(i) *= SigmoidDerivative(weightedInputs(i));
	}
	return errors;*/
}

Vector Layer::CostDerivative(Vector actualOutput, Vector expectedOutput, Vector(*CostDerivativeFunc)(Vector, Vector)) {
	return CostDerivativeFunc(actualOutput, expectedOutput);
}