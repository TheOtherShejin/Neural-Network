#pragma once

#include <chrono>
#include <algorithm>
#include <random>
#include <NeuralNetwork/Layer.h>
#include <NeuralNetwork/Maths/Vector.h>
#include <NeuralNetwork/Maths/Matrix.h>
#include <NeuralNetwork/Maths/Random.h>

#include <sstream>
#include <string>
#include <fstream>

struct DataPoint {
	Vector input, expectedOutput;
	DataPoint(Vector input, Vector expectedOutput)
		: input(input), expectedOutput(expectedOutput) {}
};
typedef std::vector<DataPoint> Dataset;

class NeuralNetwork {
private:
	void ApplyAllGradients(double learningRate, int miniBatchSize, double lambda, int datasetSize);
	void ClearAllGradients();

	void BackPropagate(DataPoint* dataPoint, Vector* actualOutput);
	int inputSize;
public:
	Cost::CostFunction* costFunction;
	std::vector<Layer> layers;

	enum MonitorType {
		MONITOR_VALIDATION_ACCURACY = 1,
		MONITOR_VALIDATION_COST = 2,
		MONITOR_TRAIN_ACCURACY = 4,
		MONITOR_TRAIN_COST = 8,
		MONITOR_SAVE_PERFORMANCE_DATA = 16
	};

	struct Settings {
		int monitorValues = MONITOR_VALIDATION_ACCURACY | MONITOR_TRAIN_ACCURACY;
		std::string performanceReportFilePath = "";
	};
	Settings settings;

	NeuralNetwork(std::vector<int> numberOfNeurons, AF::FunctionType hiddenLayerAF = AF::SigmoidAF, AF::FunctionType outputLayerAF = AF::SigmoidAF, Cost::CostType costType = Cost::MeanSquaredErrorCost);
	void RandomizeAllParameters();

	Vector Evaluate(Vector input);
	void Learn(Dataset* dataset, double learningRate, int miniBatchSize, double lambda = 0.0);
	void SGD(Dataset* dataset, int epochs, double learningRate, int miniBatchSize, double lambda = 0.0, Dataset* validation_dataset = nullptr);
	double Cost(Vector actualOutput, Vector expectedOutput, double lambda = 0.0, int datasetSize = 1);
	double RegularizationAmount(double lambda, int datasetSize) const;

	int GetInputSize() const;
	void SetActivationFunctions(AF::FunctionType hiddenLayerAF, AF::FunctionType outputLayerAF);
};