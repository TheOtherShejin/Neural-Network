#include <NeuralNetwork/NeuralNetwork.h>

NeuralNetwork::NeuralNetwork(std::vector<int> numberOfNeurons, AF::FunctionType hiddenLayerAF, AF::FunctionType outputLayerAF, Cost::CostType costType) {
	inputSize = numberOfNeurons[0];
	costFunction = Cost::GetFunctionFromEnum(costType);
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

double NeuralNetwork::Cost(Vector actualOutput, Vector expectedOutput, double lambda, int datasetSize) {
	return costFunction->Evaluate(actualOutput, expectedOutput) + RegularizationAmount(lambda, datasetSize);
}

double NeuralNetwork::RegularizationAmount(double lambda, int datasetSize) const {
	// L2 (Weight Decay) Regularization
	double weightSum = 0;
	for (auto& layer : layers) {
		for (int i = 0; i < layer.weights.rows; i++) {
			for (int j = 0; j < layer.weights.cols; j++) {
				weightSum += layer.weights.Get(i, j) * layer.weights.Get(i, j);
			}
		}
	}
	return 0.5 * weightSum * lambda / datasetSize;
}

void NeuralNetwork::BackPropagate(DataPoint* dataPoint, Vector* actualOutput) {
	Layer& outputLayer = layers[layers.size() - 1];

	// Error
	Vector errors = outputLayer.CalculateOutputLayerErrors(*actualOutput, dataPoint->expectedOutput, costFunction);
	outputLayer.UpdateGradients(errors);

	for (int i = layers.size() - 2; i >= 0; i--) {
		errors = layers[i].CalculateHiddenLayerErrors(layers[i + 1], errors);
		layers[i].UpdateGradients(errors);
	}
}

void NeuralNetwork::SGD(Dataset* dataset, int epochs, double learningRate, int miniBatchSize, double lambda, Dataset* validation_dataset) {
	float totalTimeTaken = 0.0f;
	Vector output(10);
	std::ofstream file;
	bool saveData = (settings.monitorValues & MONITOR_SAVE_PERFORMANCE_DATA);
	if (saveData) {
		file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
		try {
			file.open(settings.performanceReportFilePath);
			file << "Epoch, ";
			if (settings.monitorValues & MONITOR_TRAIN_ACCURACY) file << "Train Accuracy, ";
			if (settings.monitorValues & MONITOR_TRAIN_COST) file << "Train Cost, ";
			if (settings.monitorValues & MONITOR_VALIDATION_ACCURACY) file << "Validation Accuracy, ";
			if (settings.monitorValues & MONITOR_VALIDATION_COST) file << "Validation Cost, ";
			file << '\n';
		}
		catch (std::ofstream::failure e) {
			std::cout << e.what() << '\n';
		}
	}

	std::cout << "Training Started - " << epochs << " Epochs, Learning Rate: " << learningRate << ", Mini-Batch Size: " << miniBatchSize << ", Lambda: " << lambda << '\n';
	for (int i = 0; i < epochs; i++) {
		std::cout << "Epoch: " << i;
		if (saveData) file << i << ", ";

		// Randomize Train
		std::shuffle(dataset->begin(), dataset->end(), std::mt19937{std::random_device{}()});

		auto startTime = std::chrono::high_resolution_clock::now();
		Learn(dataset, learningRate, miniBatchSize, lambda);
		auto endTime = std::chrono::high_resolution_clock::now();
		auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

		std::cout << " - " << (dt.count() * 0.001f) << 's';
		totalTimeTaken += dt.count() * 0.001f;

		double trainCost = 0.0f;
		int correctPredictions = 0;
		int trainSize = dataset->size();
		for (int i = 0; i < trainSize; i++) {
			output = Evaluate((*dataset)[i].input);
			if (settings.monitorValues & MONITOR_TRAIN_COST) {
				trainCost += Cost(output, (*dataset)[i].expectedOutput);
			}

			int prediction = output.MaxIndex();
			if (settings.monitorValues & MONITOR_TRAIN_ACCURACY && (*dataset)[i].expectedOutput(prediction) == 1)
				correctPredictions++;
		}
		trainCost /= trainSize;
		if (settings.monitorValues & MONITOR_TRAIN_ACCURACY) {
			std::cout << " - Train Accuracy: " << correctPredictions << " / " << trainSize << " (" << (correctPredictions * 100.0f / trainSize) << "%)";
			if (saveData) file << (correctPredictions * 100.0f / trainSize) << ", ";
		}
		if (settings.monitorValues & MONITOR_TRAIN_COST) {
			std::cout << " - Train Cost : " << trainCost;
			if (saveData) file << trainCost << ", ";
		}

		if (validation_dataset == nullptr) {
			std::cout << '\n';
			if (saveData) file << '\n';
			continue;
		}

		double validationCost = 0.0f;
		correctPredictions = 0;
		int validationSize = validation_dataset->size();
		for (int i = 0; i < validationSize; i++) {
			output = Evaluate((*validation_dataset)[i].input);
			if (settings.monitorValues & MONITOR_VALIDATION_COST)
				validationCost += Cost(output, (*validation_dataset)[i].expectedOutput);

			int prediction = output.MaxIndex();
			if (settings.monitorValues & MONITOR_VALIDATION_ACCURACY && (*validation_dataset)[i].expectedOutput(prediction) == 1)
				correctPredictions++;
		}
		validationCost /= validationSize;
		if (settings.monitorValues & MONITOR_VALIDATION_ACCURACY) {
			std::cout << " - Validation Accuracy: " << correctPredictions << " / " << validationSize << " (" << (correctPredictions * 100.0f / validationSize) << "%)";
			if (saveData) file << (correctPredictions * 100.0f / validationSize) << ", ";
		}
		if (settings.monitorValues & MONITOR_VALIDATION_COST) {
			std::cout << " - Validation Cost : " << validationCost;
			if (saveData) file << validationCost << ", ";
		}
		std::cout << '\n';
		if (saveData) file << '\n';
	}
	if (saveData) {
		file.close();
		std::cout << "Performance Report Successfully Saved at: " << settings.performanceReportFilePath << '\n';
	}
	std::cout << "Training Completed in " << totalTimeTaken << "s\n";
}

void NeuralNetwork::Learn(Dataset* dataset, double learningRate, int miniBatchSize, double lambda) {
	for (int i = 0; i < (dataset->size() / miniBatchSize); i++) {
		for (int j = 0; j < miniBatchSize; j++) {
			int index = j + i * miniBatchSize;
			// Feedforward
			Vector actualOutput = Evaluate((*dataset)[index].input);

			BackPropagate(&(*dataset)[index], &actualOutput);
		}
		
		// Gradient Descent
		ApplyAllGradients(learningRate, miniBatchSize, lambda, dataset->size());
		ClearAllGradients();
	}
}

void NeuralNetwork::ApplyAllGradients(double learningRate, int miniBatchSize, double lambda, int datasetSize) {
	for (auto& layer : layers) {
		layer.ApplyGradients(learningRate, miniBatchSize, lambda, datasetSize);
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

void NeuralNetwork::SetActivationFunctions(AF::FunctionType hiddenLayerAF, AF::FunctionType outputLayerAF) {
	for (auto& layer : layers) {
		layer.SetActivationFunction(hiddenLayerAF);
	}
	layers[layers.size() - 1].SetActivationFunction(outputLayerAF);
}