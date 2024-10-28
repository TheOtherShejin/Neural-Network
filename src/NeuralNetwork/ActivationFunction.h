#pragma once

#include <math.h>
#include <vector>
#include "Vector.h"

typedef Vector (*fptr)(Vector);

namespace ActivationFunctions {
	enum FunctionType {
		LinearAF,
		BinaryStepAF,
		ReLUAF,
		LeakyReLUAF,
		SigmoidAF,
		SoftmaxAF,
		TanHAF
	};

	Vector Linear(Vector input);
	Vector BinaryStep(Vector input);
	Vector ReLU(Vector input);
	Vector LeakyReLU(Vector input);
	Vector Sigmoid(Vector input);
	Vector TanH(Vector input);
	Vector Softmax(Vector input);

	FunctionType GetFunctionEnum(Vector (*activationFunction)(Vector));
	fptr GetFunctionFromEnum(FunctionType funcType);
	fptr GetDerivativeFromEnum(FunctionType funcType);

	Vector DerivativeOf(Vector input, Vector (*activationFunction)(Vector));
	Vector LinearDerivative(Vector input);
	Vector BinaryStepDerivative(Vector input);
	Vector ReLUDerivative(Vector input);
	Vector LeakyReLUDerivative(Vector input);
	Vector SigmoidDerivative(Vector input);
	Vector TanHDerivative(Vector input);
	Vector SoftmaxDerivative(Vector input);
}
namespace AF = ActivationFunctions;