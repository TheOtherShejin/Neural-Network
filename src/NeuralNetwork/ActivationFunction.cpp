#include "ActivationFunction.h"

namespace ActivationFunctions {
	Vector Linear(Vector input) {
		return input;
	}
	Vector BinaryStep(Vector input) {
		return input.ForEach([](double element) -> double { return element >= 0; });
	}
	Vector ReLU(Vector input) {
		return input.ForEach([](double element) -> double { return fmax(0, element); });
	}
	Vector LeakyReLU(Vector input) {
		return input.ForEach([](double element) -> double { return fmax(0.1f * element, element); });
	}
	Vector Sigmoid(Vector input) {
		return input.ForEach([](double element) -> double { return 1.0f / (1.0f + exp(-element)); });
	}
	Vector TanH(Vector input) {
		return input.ForEach([](double element) -> double { return tanh(element); });
	}
	// WIP
	Vector Softmax(Vector input) {
		Vector output(input.size);
		double sum = 0;
		for (int i = 0; i < input.size; i++) {
			output(i) = exp(input(i));
			sum += output(i);
		}
		return output / sum;
	}

	FunctionType GetFunctionEnum(Vector (*activationFunction)(Vector)) {
		if (activationFunction == Sigmoid)
			return FunctionType::SigmoidAF;
		else if (activationFunction == Softmax)
			return FunctionType::SoftmaxAF;
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
		case FunctionType::SoftmaxAF:
			return Softmax;
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
	fptr GetDerivativeFromEnum(FunctionType funcType) {
		switch (funcType) {
		case FunctionType::SigmoidAF:
			return SigmoidDerivative;
			break;
		case FunctionType::SoftmaxAF:
			return SoftmaxDerivative;
			break;
		case FunctionType::TanHAF:
			return TanHDerivative;
			break;
		case FunctionType::ReLUAF:
			return ReLUDerivative;
			break;
		case FunctionType::LinearAF:
			return LinearDerivative;
			break;
		case FunctionType::BinaryStepAF:
			return BinaryStepDerivative;
			break;
		case FunctionType::LeakyReLUAF:
			return LeakyReLUDerivative;
			break;
		}
	}

	Vector DerivativeOf(Vector input, Vector (*activationFunction)(Vector)) {
		switch (GetFunctionEnum(activationFunction)) {
		case FunctionType::SigmoidAF:
			return SigmoidDerivative(input);
			break;
		case FunctionType::SoftmaxAF:
			return SoftmaxDerivative(input);
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
	Vector LinearDerivative(Vector input) {
		return Vector(input.size, 1);
	}
	Vector BinaryStepDerivative(Vector input) {
		return Vector(input.size);
	}
	Vector ReLUDerivative(Vector input) {
		return input.ForEach([](double element) -> double { return element >= 0; });
	}
	Vector LeakyReLUDerivative(Vector input) {
		return input.ForEach([](double element) -> double { return (element >= 0) ? 1 : 0.1f; });
	}
	Vector SigmoidDerivative(Vector input) {
		Vector value = Sigmoid(input);
		return value.ForEach([](double element) -> double { return element * (1 - element); });
	}
	Vector TanHDerivative(Vector input) {
		return input.ForEach([](double element) -> double { return 1.0f / pow(cosh(element), 2); });
	}
	Vector SoftmaxDerivative(Vector input) {
		Vector output = Softmax(input);
		output.ForEach([](double element) -> double { return element * (1 - element); });
		return output;
	}
}