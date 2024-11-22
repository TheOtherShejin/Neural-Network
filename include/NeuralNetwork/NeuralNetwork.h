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
	void ApplyAllGradients(double learningRate, int miniBatchSize);
	void ClearAllGradients();

	void BackPropagate(DataPoint* dataPoint, Vector* actualOutput);
	int inputSize;
public:
	std::vector<Layer> layers;
	Cost::CostFunction* costFunction;

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
	void Learn(std::vector<DataPoint> dataset, double learningRate, int miniBatchSize);
	void SGD(Dataset* dataset, int epochs, double learningRate, int miniBatchSize, Dataset* validation_dataset = nullptr);
	double Cost(Vector actualOutput, Vector expectedOutput);

	int GetInputSize() const;
	void SetActivationFunctions(AF::FunctionType hiddenLayerAF, AF::FunctionType outputLayerAF);
};