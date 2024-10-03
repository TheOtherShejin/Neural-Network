#pragma once

#include <math.h>
#include <vector>

typedef double (*fptr)(double);

namespace ActivationFunctions {
	enum FunctionType {
		LinearAF,
		BinaryStepAF,
		ReLUAF,
		LeakyReLUAF,
		SigmoidAF,
		TanHAF
	};

	double Linear(double input);
	double BinaryStep(double input);
	double ReLU(double input);
	double LeakyReLU(double input);
	double Sigmoid(double input);
	double TanH(double input);
	//std::vector<double> Softmax(std::vector<double> input); WIP

	FunctionType GetFunctionEnum(double (*activationFunction)(double));
	fptr GetFunctionFromEnum(FunctionType funcType);

	double DerivativeOf(double input, double (*activationFunction)(double));
	double LinearDerivative(double input);
	double BinaryStepDerivative(double input);
	double ReLUDerivative(double input);
	double LeakyReLUDerivative(double input);
	double SigmoidDerivative(double input);
	double TanHDerivative(double input);
}
namespace AF = ActivationFunctions;