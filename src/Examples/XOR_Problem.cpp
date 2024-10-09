#include "XOR_Problem.h"

/*
Commands:
------------
Replace the parameters in brackets with just the parameter values as shown for example:
train 10000 1.5

train [epochs] [learningRate]
test
save [saveLocation]
load [loadLocation]
quit

*/

void XOR_Problem() {
	NeuralNetwork nn({2, 3, 1}, ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid);
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
			float learningRate = std::stof(commandTokens[2]);

			std::vector<DataPoint> dataPoints = {
				{ Eigen::VectorXd{{0, 0}}, Eigen::VectorXd{{0}} },
				{ Eigen::VectorXd{{0, 1}}, Eigen::VectorXd{{1}} },
				{ Eigen::VectorXd{{1, 0}}, Eigen::VectorXd{{1}} },
				{ Eigen::VectorXd{{1, 1}}, Eigen::VectorXd{{0}} }
			};

			// Training
			std::cout << "Training Started - " << epochs << " Epochs, Learning Rate: " << learningRate << '\n';
			auto startTime = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < epochs; i++) {
				nn.Learn(dataPoints, learningRate);

				if (i % (int)round(epochs / 10.0f) == 0) {
					Eigen::VectorXd output;
					double avgCost = 0.0f;
					for (int j = 0; j < 4; j++) {
						int a = j & 1;
						int b = (j & 2) >> 1;
						output = nn.Evaluate(Eigen::VectorXd{ { (double)a, (double)b } });
						avgCost += nn.Cost(output, Eigen::VectorXd{ { (double)(a ^ b) } });
					}
					avgCost /= 4;
					std::cout << "Epoch: " << i << ", Average Cost: " << avgCost << '\n';
				}
			}
			auto endTime = std::chrono::high_resolution_clock::now();
			auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
			std::cout << "Elapsed Time For Training: " << dt.count() * 0.001f << "s\n\n";
		}
		if (commandTokens[0] == "test") {
			// Output
			Eigen::VectorXd output;
			double avgCost = 0.0f;
			std::cout << "XOR Problem Example:\n";
			for (int i = 0; i < 4; i++) {
				int a = i & 1;
				int b = (i & 2) >> 1;
				output = nn.Evaluate(Eigen::VectorXd{ { (double)a, (double)b } });
				avgCost += nn.Cost(output, Eigen::VectorXd{ { (double)(a ^ b) } });
				std::cout << (output[0] > 0.5f ? 1 : 0) << ' ' << output[0] << '\n';
			}
			avgCost /= 4;
			std::cout << "Average Cost: " << avgCost << "\n\n";

		}
		else if (commandTokens[0] == "load") {
			nn = LoadModelFromCSV(commandTokens[1]);
			std::cout << '\n';
		}
		else if (commandTokens[0] == "save") {
			SaveModelToCSV(commandTokens[1], &nn);
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
				"train 10000 1.5\n\n"
				"train [epochs] [learningRate]\n"
				"test\n"
				"save [saveLocation]\n"
				"load [loadLocation]\n"
				"quit\n\n";
		}
	}
}