#include <Examples/Digit_Classifier.h>

void DigitClassifierApp::Run() {
	Init();
	Update();
}

void DigitClassifierApp::Init() {
	Log("Loading Training Dataset...\n");
	train_dataset = LoadIntoDataset("datasets/mnist_train_normalized.csv", 0.15, &validation_dataset);
	Log("Loading Test Dataset...\n");
	test_dataset = LoadIntoDataset("datasets/mnist_test_normalized.csv");

	nn.monitorValues = NeuralNetwork::MONITOR_TRAIN_ACCURACY | NeuralNetwork::MONITOR_VALIDATION_ACCURACY;
}

void DigitClassifierApp::Update() {
	Log("Enter help to get help.\n");
	std::string command;
	while (runProgram) {
		Log("Enter a command: ");
		std::getline(std::cin, command);
		if (command == "") continue;

		StringTokens commandTokens = Tokenize(command, ' ');

		if (commandTokens[0] == "train") Train(std::stoi(commandTokens[1]), std::stod(commandTokens[2]), std::stoi(commandTokens[3]));
		else if (commandTokens[0] == "test") Test(commandTokens[1] == "random" ? true : false);
		else if (commandTokens[0] == "load") Load(commandTokens[1]);
		else if (commandTokens[0] == "save") Save(commandTokens[1], commandTokens[2]);
		else if (commandTokens[0] == "quit") Quit();
		else if (commandTokens[0] == "reset") Reset();
		else if (commandTokens[0] == "help") Help();
	}
}

void DigitClassifierApp::Train(int epochs, double learningRate, int miniBatchSize) {
	nn.SGD(&train_dataset, epochs, learningRate, miniBatchSize, &validation_dataset);
}
void DigitClassifierApp::Test(bool random) {
	if (!random) {
		// Output
		Log("Testing...\n");
		Vector output(10);
		double cost = 0.0f;
		int correctPredictions = 0;
		for (int i = 0; i < test_dataset.size(); i++) {
			output = nn.Evaluate(test_dataset[i].input);
			cost += nn.Cost(output, test_dataset[i].expectedOutput);
			int prediction = output.MaxIndex();
			if (test_dataset[i].expectedOutput(prediction) == 1) correctPredictions++;
		}
		cost /= test_dataset.size();
		Log("Test Completed - Accuracy: " + std::to_string(correctPredictions) + " / " + std::to_string(test_dataset.size()) + " (" + std::to_string(correctPredictions / 100.0f)
			+ "%) - Cost: " + std::to_string(cost) + "\n\n");
	}
	else { // Random
		int index = RandomFromRange(0, test_dataset.size() - 1);
		Vector output = nn.Evaluate(test_dataset[index].input);

		int actual = test_dataset[index].expectedOutput.MaxIndex();
		int prediction = output.MaxIndex();
		Log("Actual Digit: " + actual + '\n');
		output.Print();
		Log("Prediction: " + std::to_string(prediction) + "\n\n");
	}
}
void DigitClassifierApp::Load(std::string path) {
	nn = LoadModelFromCSV(path);
}
void DigitClassifierApp::Save(std::string format, std::string path) {
	if (format == "csv") SaveModelToCSV(path, &nn);
	else if (format == "js") SaveModelToJS(path, &nn);
}
void DigitClassifierApp::Quit() {
	runProgram = false;
}
void DigitClassifierApp::Reset() {
	nn.RandomizeAllParameters();
}
void DigitClassifierApp::Help() {
	Log(
		"Commands:\n"
		"------------\n"
		"Replace the parameters in brackets with just the parameter values as shown for example:\n"
		"train 30 3 10\n\n"
		"train [epochs] [learningRate] [miniBatchSize]\n"
		"test [random / all]\n"
		"save [format] [saveLocation] - Available Formats: .csv and .js\n"
		"load [loadLocation]\n"
		"reset - Randomize the neural network's parameters."
		"help - Show these instructions."
		"quit\n\n"
	);
}