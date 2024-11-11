#include <NeuralNetwork/NeuralNetwork.h>

NeuralNetwork::NeuralNetwork(std::vector<int> numberOfNeurons, Vector (*hiddenLayerAF)(Vector), Vector (*outputLayerAF)(Vector), double (*CostFunction)(Vector, Vector)) {
	inputSize = numberOfNeurons[0];
	this->CostFunction = CostFunction;
	CostFuncDerivative = Cost::GetDerivativeFromEnum(Cost::GetFunctionEnum(CostFunction));
	for (int i = 1; i < numberOfNeurons.size(); i++) {
		layers.push_back(Layer(numberOfNeurons[i], numberOfNeurons[i-1], hiddenLayerAF));
	}
	layers[layers.size()-1].SetActivationFunction(outputLayerAF);
}

void NeuralNetwork::RandomizeAllParameters() {
	for (auto& layer : layers) {
		layer.RandomizeParameters();
	}
}

Vector NeuralNetwork::Evaluate(Vector input) {
	for (int i = 0; i < layers.size(); i++) {
		input = layers[i].FeedForward(input);
	}
	return input;
}

double NeuralNetwork::Cost(Vector actualOutput, Vector expectedOutput) {
	return (actualOutput - expectedOutput).MagnitudeSqr() * 0.5;
}

void NeuralNetwork::BackPropagate(DataPoint* dataPoint, Vector* actualOutput) {
	Layer& outputLayer = layers[layers.size() - 1];

	// Error
	Vector errors = outputLayer.CalculateOutputLayerErrors(*actualOutput, dataPoint->expectedOutput, CostFuncDerivative);
	outputLayer.UpdateGradients(errors);

	for (int i = layers.size() - 2; i >= 0; i--) {
		errors = layers[i].CalculateHiddenLayerErrors(layers[i + 1], errors);
		layers[i].UpdateGradients(errors);
	}
}

void NeuralNetwork::SGD(Dataset* dataset, int epochs, double learningRate, int miniBatchSize, Dataset* validation_dataset) {
	float totalTimeTaken = 0.0f;
	Vector output(10);
	Log("Training Started - " + std::to_string(epochs) + " Epochs, Learning Rate: " + std::to_string(learningRate) + ", Mini-Batch Size: " + std::to_string(miniBatchSize) + '\n');

	for (int i = 0; i < epochs; i++) {
		Log("Epoch: " + i);

		// Randomize Train
		std::shuffle(dataset->begin(), dataset->end(), std::mt19937{std::random_device{}()});

		auto startTime = std::chrono::high_resolution_clock::now();
		Learn(*dataset, learningRate, miniBatchSize);
		auto endTime = std::chrono::high_resolution_clock::now();
		auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

		std::cout << " - " << (dt.count() * 0.001f) << 's';
		totalTimeTaken += dt.count() * 0.001f;

		double trainCost = 0.0f;
		int correctPredictions = 0;
		int trainSize = dataset->size();
		for (int i = 0; i < trainSize; i++) {
			output = Evaluate((*dataset)[i].input);
			if (monitorValues & MONITOR_TRAIN_COST)
				trainCost += Cost(output, (*dataset)[i].expectedOutput);

			int prediction = output.MaxIndex();
			if (monitorValues & MONITOR_TRAIN_ACCURACY && (*dataset)[i].expectedOutput(prediction) == 1)
				correctPredictions++;
		}
		trainCost /= trainSize;
		if (monitorValues & MONITOR_VALIDATION_ACCURACY)
			Log(" - Train Accuracy: " + std::to_string(correctPredictions) + " / " + std::to_string(trainSize) + " (" + std::to_string(correctPredictions * 100.0f / trainSize) + "%)");
		if (monitorValues & MONITOR_VALIDATION_COST)
			Log(" - Train Cost : " + std::to_string(trainCost));

		if (validation_dataset == nullptr) {
			Log("\n");
			continue;
		}

		double validationCost = 0.0f;
		correctPredictions = 0;
		int validationSize = validation_dataset->size();
		for (int i = 0; i < validationSize; i++) {
			output = Evaluate((*validation_dataset)[i].input);
			if (monitorValues & MONITOR_VALIDATION_COST)
				validationCost += Cost(output, (*validation_dataset)[i].expectedOutput);

			int prediction = output.MaxIndex();
			if (monitorValues & MONITOR_VALIDATION_ACCURACY && (*validation_dataset)[i].expectedOutput(prediction) == 1)
				correctPredictions++;
		}
		validationCost /= validationSize;
		if (monitorValues & MONITOR_VALIDATION_ACCURACY)
			Log(" - Validation Accuracy: " + std::to_string(correctPredictions) + " / " + std::to_string(validationSize) + " (" + std::to_string(correctPredictions * 100.0f / validationSize) + "%)");
		if (monitorValues & MONITOR_VALIDATION_COST)
			Log(" - Validation Cost : " + std::to_string(validationCost));
		Log("\n");
	}
	Log("Training Completed in " + std::to_string(totalTimeTaken) + "s\n");
}

void NeuralNetwork::Learn(std::vector<DataPoint> dataset, double learningRate, int miniBatchSize) {
	for (int i = 0; i < (dataset.size() / miniBatchSize); i++) {
		for (int j = 0; j < miniBatchSize; j++) {
			int index = j + i * miniBatchSize;
			// Feedforward
			Vector actualOutput = Evaluate(dataset[index].input);

			BackPropagate(&dataset[index], &actualOutput);
		}

		// Gradient Descent
		ApplyAllGradients(learningRate, miniBatchSize);
		ClearAllGradients();
	}
}

void NeuralNetwork::ApplyAllGradients(double learningRate, int miniBatchSize) {
	for (auto& layer : layers) {
		layer.ApplyGradients(learningRate, miniBatchSize);
	}
}

void NeuralNetwork::ClearAllGradients() {
	for (auto& layer : layers) {
		layer.ClearGradients();
	}
}

int NeuralNetwork::GetInputSize() const {
	return inputSize;
}

void NeuralNetwork::SetActivationFunctions(Vector (*hiddenLayerAF)(Vector), Vector (*outputLayerAF)(Vector)) {
	for (auto& layer : layers) {
		layer.SetActivationFunction(hiddenLayerAF);
	}
	layers[layers.size() - 1].SetActivationFunction(outputLayerAF);
}