#pragma once

#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <NeuralNetwork/Layer.h>
#include <NeuralNetwork/Maths/Vector.h>
#include <NeuralNetwork/Maths/Matrix.h>

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
	
	enum MonitorType {
		MONITOR_VALIDATION_ACCURACY = 1,
		MONITOR_VALIDATION_COST = 2,
		MONITOR_TRAIN_ACCURACY = 4,
		MONITOR_TRAIN_COST = 8
	};
	int monitorValues = MONITOR_VALIDATION_ACCURACY | MONITOR_TRAIN_ACCURACY;

	NeuralNetwork(std::vector<int> numberOfNeurons, Vector (*hiddenLayerAF)(Vector) = AF::Sigmoid, Vector (*outputLayerAF)(Vector) = AF::Sigmoid, double (*CostFunction)(Vector, Vector) = Cost::MeanSquaredError);
	void RandomizeAllParameters();

	Vector Evaluate(Vector input);
	void Learn(std::vector<DataPoint> dataset, double learningRate, int miniBatchSize);
	void SGD(Dataset* dataset, int epochs, double learningRate, int miniBatchSize, Dataset* validation_dataset = nullptr);
	double Cost(Vector actualOutput, Vector expectedOutput);

	int GetInputSize() const;
	void SetActivationFunctions(Vector (*hiddenLayerAF)(Vector), Vector (*outputLayerAF)(Vector));
};