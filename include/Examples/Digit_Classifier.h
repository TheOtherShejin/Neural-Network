#pragma once

#include <Examples/Application.h>
#include <Examples/MNIST_Processor.h>
#include <Examples/Utils.h>
#include <NeuralNetwork/ModelExporter.h>
#include <chrono>
#include <algorithm>
#include <random>

class DigitClassifierApp : public Application {
private:
	bool runProgram = true;
	double lambda = 0.0;
	NeuralNetwork nn{ { 784, 30, 10 }, ActivationFunctions::SigmoidAF, ActivationFunctions::SoftmaxAF,  Cost::CategoricalCrossEntropyCost };
	Dataset train_dataset, validation_dataset, test_dataset;
	void Init();
	void Update();

	void Train(int epochs, double learningRate, int miniBatchSize, double lambda);
	void Test(bool random);
	void Load(std::string path);
	void Save(std::string format, std::string path);
	void TogglePerformanceReport(bool enable, std::string path);
	void Quit();
	void Reset();
	void Clear();
	void Help();
};