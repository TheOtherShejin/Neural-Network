#include <Examples/LearningSlowDown.h>

void LearningSlowDownApp::Run() {
	Init();
	Update();
}

void LearningSlowDownApp::Init() {
	nn1.settings.monitorValues = NeuralNetwork::MONITOR_TRAIN_COST | NeuralNetwork::MONITOR_SAVE_PERFORMANCE_DATA;
	nn2.settings.monitorValues = nn1.settings.monitorValues;

	dataset.push_back(DataPoint(Vector(1, 1), Vector(1, 0)));
}

void LearningSlowDownApp::Update() {
	nn1.settings.performanceReportFilePath = "reports/learningSlowDown11.csv";
	nn1.layers[0].weights(0, 0) = 0.6;
	nn1.layers[0].biases(0) = 0.9;
	nn1.SGD(&dataset, 300, 0.15, 1);

	nn1.settings.performanceReportFilePath = "reports/learningSlowDown12.csv";
	nn1.layers[0].weights(0, 0) = 2.0;
	nn1.layers[0].biases(0) = 2.0;
	nn1.SGD(&dataset, 300, 0.15, 1);

	nn2.settings.performanceReportFilePath = "reports/learningSlowDown21.csv";
	nn2.layers[0].weights(0, 0) = 0.6;
	nn2.layers[0].biases(0) = 0.9;
	nn2.SGD(&dataset, 300, 0.025, 1);

	nn2.settings.performanceReportFilePath = "reports/learningSlowDown22.csv";
	nn2.layers[0].weights(0, 0) = 2.0;
	nn2.layers[0].biases(0) = 2.0;
	nn2.SGD(&dataset, 300, 0.025, 1);
}