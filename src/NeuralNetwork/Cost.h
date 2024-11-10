#pragma once

#include "Vector.h"

namespace Cost {
	typedef double (*cfptr)(Vector, Vector);
	typedef Vector (*dfptr)(Vector, Vector);

	enum CostType {
		MeanSquaredErrorCost,
		BinaryCrossEntropyCost,
		CategoricalCrossEntropyCost
	};

	CostType GetFunctionEnum(double (*costFunc)(Vector, Vector));
	cfptr GetFunctionFromEnum(CostType funcType);
	dfptr GetDerivativeFromEnum(CostType funcType);

	double MeanSquaredError(Vector actualOutput, Vector expectedOutput);
	double BinaryCrossEntropy(Vector actualOutput, Vector expectedOutput);
	double CategoricalCrossEntropy(Vector acutalOutput, Vector expectedOutput);

	Vector MeanSquaredErrorDerivative(Vector actualOutput, Vector expectedOutput);
	Vector BinaryCrossEntropyDerivative(Vector actualOutput, Vector expectedOutput);
	Vector CategoricalCrossEntropyDerivative(Vector actualOutput, Vector expectedOutput);
}