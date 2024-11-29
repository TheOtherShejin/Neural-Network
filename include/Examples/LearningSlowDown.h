#pragma once

#include <NeuralNetwork/NeuralNetwork.h>

class LearningSlowDownApp {
public:
	void Run();
private:
	Dataset dataset;
	NeuralNetwork nn1{ {1, 1}, AF::SigmoidAF, AF::SigmoidAF, Cost::MeanSquaredErrorCost };
	NeuralNetwork nn2{ {1, 1}, AF::SigmoidAF, AF::SigmoidAF, Cost::SigmoidCrossEntropyCost };

	void Init();
	void Update();
};