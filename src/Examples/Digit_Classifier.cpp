#include <Examples/Digit_Classifier.h>

void DigitClassifierApp::Init() {
	std::cout << "Loading Training Dataset...\n";
	train_dataset = LoadIntoDataset("datasets/mnist_train_normalized.csv", 0.15, &validation_dataset);
	std::cout << "Loading Test Dataset...\n";
	test_dataset = LoadIntoDataset("datasets/mnist_test_normalized.csv");

	nn.settings.monitorFlags = NeuralNetwork::MONITOR_TRAIN_ACCURACY | NeuralNetwork::MONITOR_VALIDATION_ACCURACY | NeuralNetwork::MONITOR_SAVE_PERFORMANCE_DATA;
	nn.settings.performanceReportFilePath = "reports/performance.csv";
}

void DigitClassifierApp::Update() {
	std::cout << "Enter help to get help.\n";
	std::string command;
	while (runProgram) {
		std::cout << "Enter a command: ";
		std::getline(std::cin, command);
		if (command == "") continue;

		StringTokens commandTokens = Tokenize(command, ' ');

		if (commandTokens[0] == "train") Train(std::stoi(commandTokens[1]), std::stod(commandTokens[2]), std::stoi(commandTokens[3]), std::stod(commandTokens[4]));
		else if (commandTokens[0] == "test") Test(commandTokens[1] == "random" ? true : false);
		else if (commandTokens[0] == "load") Load(commandTokens[1]);
		else if (commandTokens[0] == "save") Save(commandTokens[1], commandTokens[2]);
		else if (commandTokens[0] == "performanceReport") TogglePerformanceReport(std::stoi(commandTokens[1]), commandTokens[2]);
		else if (commandTokens[0] == "quit") Quit();
		else if (commandTokens[0] == "reset") Reset();
		else if (commandTokens[0] == "clear") Clear();
		else if (commandTokens[0] == "help") Help();
	}
}

void DigitClassifierApp::Train(int epochs, double learningRate, int miniBatchSize, double lambda) {
	this->lambda = lambda;
	nn.SGD(&train_dataset, epochs, learningRate, miniBatchSize, lambda, true, &validation_dataset);
}
void DigitClassifierApp::Test(bool random) {
	Vector output(10);
	if (!random) {
		// Output
		std::cout << "Testing...\n";
		double cost = 0.0f;
		int correctPredictions = 0;
		for (auto& datapoint : test_dataset) {
			output = nn.Evaluate(datapoint.input);
			cost += nn.Cost(output, datapoint.expectedOutput);
			int prediction = output.MaxIndex();
			if (datapoint.expectedOutput(prediction) == 1) correctPredictions++;
		}
		cost /= test_dataset.size();
		std::cout << "Test Completed - Accuracy: " << correctPredictions << " / " << test_dataset.size() << " (" << (correctPredictions / 100.0f)
			<< "%) - Cost: " << cost << "\n\n";
	}
	else { // Random
		int index = RandomFromRange(0, test_dataset.size() - 1);
		output = nn.Evaluate(test_dataset[index].input);

		int actual = test_dataset[index].expectedOutput.MaxIndex();
		int prediction = output.MaxIndex();
		std::cout << "Actual Digit : " << actual << '\n';
		output.Print();
		std::cout << "Prediction: " << std::to_string(prediction) << "\n\n";
	}
}
void DigitClassifierApp::Load(std::string path) {
	nn = LoadModelFromCSV(path);
}
void DigitClassifierApp::Save(std::string format, std::string path) {
	if (format == "csv") SaveModelToCSV(path, &nn);
	else if (format == "js") SaveModelToJS(path, &nn);
}
void DigitClassifierApp::TogglePerformanceReport(bool enable, std::string path) {
	if (enable) {
		nn.settings.monitorFlags |= NeuralNetwork::MONITOR_SAVE_PERFORMANCE_DATA;
		nn.settings.performanceReportFilePath = path;
	}
	else nn.settings.monitorFlags = nn.settings.monitorFlags & 0b01111;
}

void DigitClassifierApp::Quit() {
	runProgram = false;
}
void DigitClassifierApp::Reset() {
	nn.RandomizeAllParameters();
}
void DigitClassifierApp::Clear() {
	system("cls");
}
void DigitClassifierApp::Help() {
	std::cout <<
		"Commands:\n"
		"------------\n"
		"Replace the parameters in brackets with just the parameter values as shown for example:\n"
		"train 30 3 10\n\n"
		"train [epochs] [learningRate] [miniBatchSize] [lambda]\n"
		"test [random / all]\n"
		"save [format] [saveLocation] - Available Formats: .csv and .js\n"
		"load [loadLocation]\n"
		"reset - Randomize the neural network's parameters.\n"
		"help - Show these instructions.\n"
		"clear - Clear the console.\n"
		"performanceReport [enable: 1 (true) / 0 (false)] [reportSavePath]\n"
		"quit\n\n"
	;
}