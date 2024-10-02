#include <iostream>
#include <chrono>
#include "NeuralNetwork.h"
#include "ModelExporter.h"

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

int main() {
	NeuralNetwork nn({ 2, 3, 1 }, ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid);
	bool runProgram = true;
	while (runProgram) {
		std::string command;
		std::cout << "Enter a command:\n";
		std::getline(std::cin, command);

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
			{ {0, 0}, {0} },
			{ {0, 1}, {1} },
			{ {1, 0}, {1} },
			{ {1, 1}, {0} }
			};

			// Training
			std::cout << "Training Started - " << epochs << " Epochs, Learning Rate: " << learningRate << '\n';
			auto startTime = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < epochs; i++) {
				nn.Learn(dataPoints, learningRate);

				if (i % (int)(epochs/10) == 0) {
					std::vector<double> output;
					double avgLoss = 0.0f;
					for (int j = 0; j < 4; j++) {
						int a = j & 1;
						int b = (j & 2) >> 1;
						output = nn.CalculateOutput({ (double)a, (double)b });
						avgLoss += nn.Loss(output, { (double)(a ^ b) });
					}
					avgLoss /= 4;
					std::cout << "Epoch: " << i << ", Average Loss: " << avgLoss << '\n';
				}
			}
			auto endTime = std::chrono::high_resolution_clock::now();
			auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
			std::cout << "Elapsed Time For Training: " << dt.count() * 0.001f << "s\n\n";
		}
		else if (commandTokens[0] == "test") {
			// Output
			std::vector<double> output;
			double avgLoss = 0.0f;
			std::cout << "XOR Problem Example:\n";
			for (int i = 0; i < 4; i++) {
				int a = i & 1;
				int b = (i & 2) >> 1;
				output = nn.CalculateOutput({ (double)a, (double)b });
				avgLoss += nn.Loss(output, { (double)(a ^ b) });
				std::cout << (output[0] > 0.5f ? 1 : 0) << ' ' << output[0] << '\n';
			}
			avgLoss /= 4;
			std::cout << "Average Loss: " << avgLoss << "\n\n";

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
	}

	return 0;
}