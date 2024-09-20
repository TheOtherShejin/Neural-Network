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
	// WIP
	/*std::vector<double> Softmax(std::vector<double> input) {
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
	}*/

	FunctionType GetFunctionEnum(double (*activationFunction)(double)) {
		if (activationFunction == Sigmoid)
			return FunctionType::SigmoidAF;
		else if (activationFunction == TanH)
			return FunctionType::TanHAF;
		else if (activationFunction == ReLU)
			return FunctionType::ReLUAF;
		else if (activationFunction == Linear)
			return FunctionType::LinearAF;
		else if (activationFunction == BinaryStep)
			return FunctionType::BinaryStepAF;
		else if (activationFunction == LeakyReLU)
			return FunctionType::LeakyReLUAF;
	}
	fptr GetFunctionFromEnum(FunctionType funcType) {
		switch (funcType) {
		case FunctionType::SigmoidAF:
			return Sigmoid;
			break;
		case FunctionType::TanHAF:
			return TanH;
			break;
		case FunctionType::ReLUAF:
			return ReLU;
			break;
		case FunctionType::LinearAF:
			return Linear;
			break;
		case FunctionType::BinaryStepAF:
			return BinaryStep;
			break;
		case FunctionType::LeakyReLUAF:
			return LeakyReLU;
			break;
		}
	}

	double DerivativeOf(double input, double (*activationFunction)(double)) {
		switch (GetFunctionEnum(activationFunction)) {
		case FunctionType::SigmoidAF:
			return SigmoidDerivative(input);
			break;
		case FunctionType::TanHAF:
			return TanHDerivative(input);
			break;
		case FunctionType::ReLUAF:
			return ReLUDerivative(input);
			break;
		case FunctionType::LinearAF:
			return LinearDerivative(input);
			break;
		case FunctionType::BinaryStepAF:
			return BinaryStepDerivative(input);
			break;
		case FunctionType::LeakyReLUAF:
			return LeakyReLUDerivative(input);
			break;
		}
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