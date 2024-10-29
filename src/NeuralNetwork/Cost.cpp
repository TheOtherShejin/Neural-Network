#include "Cost.h"

namespace Cost {
	CostType GetFunctionEnum(double (*costFunc)(Vector, Vector)) {
		if (costFunc == MeanSquaredError)
			return CostType::MeanSquaredErrorCost;
		else if (costFunc == CrossEntropy)
			return CostType::CrossEntropyCost;
	}
	cfptr GetFunctionFromEnum(CostType funcType) {
		switch (funcType) {
		case CostType::MeanSquaredErrorCost:
			return MeanSquaredError;
			break;
		case CostType::CrossEntropyCost:
			return CrossEntropy;
			break;
		}
	}
	dfptr GetDerivativeFromEnum(CostType funcType) {
		switch (funcType) {
		case CostType::MeanSquaredErrorCost:
			return MeanSquaredErrorDerivative;
			break;
		case CostType::CrossEntropyCost:
			return CrossEntropyDerivative;
			break;
		}
	}

	double MeanSquaredError(Vector actualOutput, Vector expectedOutput) {
		return (actualOutput - expectedOutput).MagnitudeSqr() * 0.5;
	}
	double CrossEntropy(Vector actualOutput, Vector expectedOutput) {
		double cost = 0;
		for (int i = 0; i < actualOutput.size; i++) {
			cost += -expectedOutput(i) * log(actualOutput(i)) - (1 - expectedOutput(i)) * log(1 - actualOutput(i));
		}
		return cost;
	}

	Vector MeanSquaredErrorDerivative(Vector actualOutput, Vector expectedOutput) {
		return actualOutput - expectedOutput;
	}
	Vector CrossEntropyDerivative(Vector actualOutput, Vector expectedOutput) {
		Vector output(actualOutput.size);
		for (int i = 0; i < output.size; i++) {
			output(i) = (actualOutput(i) - expectedOutput(i)) / (actualOutput(i) * (1 - actualOutput(i)));
		}
		return output;
	}
}