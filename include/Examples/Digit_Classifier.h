#pragma once

#include "MNIST_Processor.h"
#include <NeuralNetwork/ModelExporter.h>
#include "Utils.h"
#include <chrono>
#include <algorithm>
#include <random>

class DigitClassifierApp {
public:
	void Run();
private:
	bool runProgram = true;
	double lambda = 5.0;
	NeuralNetwork nn{ { 784, 30, 10 }, ActivationFunctions::SigmoidAF, ActivationFunctions::SoftmaxAF,  Cost::CategoricalCrossEntropyCost };
	Dataset train_dataset, validation_dataset, test_dataset;
	void Init();
	void Update();

	void Train(int epochs, double learningRate, int miniBatchSize);
	void Test(bool random);
	void Load(std::string path);
	void Save(std::string format, std::string path);
	void TogglePerformanceReport(bool enable, std::string path);
	void Quit();
	void Reset();
	void Help();
};