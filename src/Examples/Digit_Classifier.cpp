#include "Digit_Classifier.h"

void Digit_Classifier() {
	NeuralNetwork nn({ 784, 30, 10 }, ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid, Cost::CrossEntropy);
	std::cout << "Loading Training Dataset...\n";
	std::vector<DataPoint> train_dataset = LoadIntoDataset("datasets/mnist_train_normalized.csv");
	std::cout << "Loading Test Dataset...\n";
	std::vector<DataPoint> test_dataset = LoadIntoDataset("datasets/mnist_test_normalized.csv");

	bool runProgram = true;
	std::cout << "Enter help to get help.\n";
	while (runProgram) {
		std::string command;
		std::cout << "Enter a command:\n";
		std::getline(std::cin, command);
		if (command == "") continue;

		std::stringstream commandSS(command);
		std::vector<std::string> commandTokens;
		std::string token;
		while (std::getline(commandSS, token, ' ')) {
			commandTokens.push_back(token);
		}

		if (commandTokens[0] == "train") {
			int epochs = std::stoi(commandTokens[1]);
			double learningRate = std::stod(commandTokens[2]);
			int miniBatchSize = std::stoi(commandTokens[3]);

			// Training
			nn.SGD(&train_dataset, epochs, learningRate, miniBatchSize);
		}
		if (commandTokens[0] == "test") {
			// Output
			std::cout << "Testing...\n";
			Vector output(10);
			double cost = 0.0f;
			int correctPredictions = 0;
			for (int i = 0; i < test_dataset.size(); i++) {
				output = nn.Evaluate(test_dataset[i].input);
				cost += nn.Cost(output, test_dataset[i].expectedOutput);

				int prediction = 0;
				double bestConfidence = output(0);
				for (int i = 0; i < 10; i++) {
					if (output(i) > bestConfidence) {
						bestConfidence = output(i);
						prediction = i;
					}
				}

				if (test_dataset[i].expectedOutput(prediction) == 1) correctPredictions++;
			}
			cost /= test_dataset.size();
			std::cout << "Test Completed - Accuracy: " << correctPredictions << " / 10000 (" << (correctPredictions / 100.0f) << "%) - Cost: " << cost << '\n';
		}
		else if (commandTokens[0] == "load") {
			nn = LoadModelFromCSV(commandTokens[1]);
			std::cout << '\n';
		}
		else if (commandTokens[0] == "save") {
			std::string format = commandTokens[1];
			if (format == "csv") SaveModelToCSV(commandTokens[2], &nn);
			else if (format == "js") SaveModelToJS(commandTokens[2], &nn);
			std::cout << '\n';
		}
		else if (commandTokens[0] == "quit") {
			runProgram = false;
		}
		else if (commandTokens[0] == "reset") {
			nn.RandomizeAllParameters();
		}
		else if (commandTokens[0] == "help") {
			std::cout <<
				"Commands:\n"
				"------------\n"
				"Replace the parameters in brackets with just the parameter values as shown for example:\n"
				"train 30 3 10\n\n"
				"train [epochs] [learningRate] [miniBatchSize]\n"
				"test\n"
				"save [format] [saveLocation]\n"
				"load [loadLocation]\n"
				"reset"
				"quit\n\n";
		}
	}
}