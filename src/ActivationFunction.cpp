#include "ActivationFunction.h"

namespace ActivationFunctions {
	double Linear(double input) {
		return input;
	}
	double BinaryStep(double input) {
		return (input >= 0);
	}
	double ReLU(double input) {
		return fmax(0, input);
	}
	double LeakyReLU(double input) {
		return fmax(0.1f * input, input);
	}
	double Sigmoid(double input) {
		return 1.0f / (1 + exp(-input));
	}
	double TanH(double input) {
		return tanh(input);
	}
	std::vector<double> Softmax(std::vector<double> input) {
		std::vector<double> output;
		double sum = 0;
		for (int i = 0; i < input.size(); i++) {
			output[i] = exp(input[i]);
			sum += output[i];
		}
		for (int i = 0; i < input.size(); i++) {
			output[i] /= sum;
		}
		return output;
	}

	double DerivativeOf(double input, double (*activationFunction)(double)) {
		if (activationFunction == Sigmoid)
			return SigmoidDerivative(input);
		else if (activationFunction == TanH)
			return TanHDerivative(input);
		else if (activationFunction == Linear)
			return LinearDerivative(input);
		else if (activationFunction == BinaryStep)
			return BinaryStepDerivative(input);
		else if (activationFunction == ReLU)
			return ReLUDerivative(input);
		else if (activationFunction == LeakyReLU)
			return LeakyReLUDerivative(input);
	}
	double LinearDerivative(double input) {
		return 1;
	}
	double BinaryStepDerivative(double input) {
		return 0;
	}
	double ReLUDerivative(double input) {
		return (input >= 0);
	}
	double LeakyReLUDerivative(double input) {
		if (input >= 0) return 1;
		else return 0.1f;
	}
	double SigmoidDerivative(double input) {
		double value = Sigmoid(input);
		return value * (1 - value);
	}
	double TanHDerivative(double input) {
		return 1.0f / pow((cosh(input)), 2);
	}
}