/*#include "MNIST_Processor.h"

void LoadNormalizeAndSave(std::string path, std::string savePath) {
	std::ifstream rawFile;
	std::ofstream normalizedFile;

	rawFile.open(path);
	normalizedFile.open(savePath);

	if (!rawFile.good() || !normalizedFile.good()) return;

	std::string dataPoint;
	std::string subString;
	while (std::getline(rawFile, dataPoint)) {
		std::stringstream dataPointSS(dataPoint);
		std::getline(dataPointSS, subString, ',');
		normalizedFile << subString << ',';
		while (std::getline(dataPointSS, subString, ',')) {
			normalizedFile << (std::stod(subString) / 255.0) << ',';
		}
		normalizedFile << '\n';
	}

	rawFile.close();
	normalizedFile.close();
	std::cout << "MNIST data has been normalized from: " << path << ", to: " << savePath << '\n';
}

std::vector<DataPoint> LoadIntoDataset(std::string path) {
	std::ifstream file;
	std::vector<DataPoint> dataset;
	file.open(path);

	if (!file.good()) {
		std::cout << "Failed to open file at path: " << path << '\n';
		return dataset;
	}

	std::string line, substring;
	std::vector<double> input = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	std::vector<double> expectedOutput;
	std::stringstream ss(line);
	while (std::getline(file, line)) {
		ss = std::stringstream(line);
		expectedOutput.clear();
		std::getline(ss, substring, ',');
		int number = std::stoi(substring);
		input[number] = 1;
		while (std::getline(ss, substring, ',')) {
			expectedOutput.push_back(std::stod(substring));
		}
		dataset.push_back(DataPoint(input, expectedOutput));
		input[number] = 0;
	}
}*/