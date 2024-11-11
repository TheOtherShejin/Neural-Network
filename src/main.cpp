#include <Examples/Digit_Classifier.h>

int main() {
	DigitClassifierApp app;

	try {
		app.Run();
	}
	catch (std::exception e) {
		std::cout << e.what() << '\n';
		return 1;
	}

	return 0;
}