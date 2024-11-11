#include <Examples/MNIST_Processor.h>

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

std::vector<DataPoint> LoadIntoDataset(std::string path, float validationSplit, std::vector<DataPoint>* validation_dataset) {
	std::ifstream file;
	std::vector<DataPoint> dataset;
	file.open(path);

	if (!file.good()) {
		std::cout << "Failed to open file at path: " << path << '\n';
		return dataset;
	}

	std::string line, substring;
	Vector input(784);
	input.SetZero();
	Vector expectedOutput{ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0} };
	std::stringstream ss(line);
	int imageIndex = 0;
	int trainAmount = (1 - validationSplit) * 60000;
	while (std::getline(file, line)) {
		ss = std::stringstream(line);
		std::getline(ss, substring, ',');
		int number = std::stoi(substring);
		expectedOutput(number) = 1;

		int index = 0;
		while (std::getline(ss, substring, ',')) {
			input(index) = std::stod(substring);
			index++;
		}

		if (imageIndex < trainAmount)
			dataset.push_back(DataPoint(input, expectedOutput));
		else {
			if (validation_dataset != nullptr)
			validation_dataset->push_back(DataPoint(input, expectedOutput));
		}
		expectedOutput(number) = 0;
		imageIndex++;
	}
	return dataset;
}