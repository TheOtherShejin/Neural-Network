#pragma once

#include "Vector.h"

namespace Cost {
	typedef double (*cfptr)(Vector, Vector);
	typedef Vector (*dfptr)(Vector, Vector);

	enum CostType {
		MeanSquaredErrorCost,
		CrossEntropyCost
	};

	CostType GetFunctionEnum(double (*costFunc)(Vector, Vector));
	cfptr GetFunctionFromEnum(CostType funcType);
	dfptr GetDerivativeFromEnum(CostType funcType);

	double MeanSquaredError(Vector actualOutput, Vector expectedOutput);
	double CrossEntropy(Vector actualOutput, Vector expectedOutput);

	Vector MeanSquaredErrorDerivative(Vector actualOutput, Vector expectedOutput);
	Vector CrossEntropyDerivative(Vector actualOutput, Vector expectedOutput);
}