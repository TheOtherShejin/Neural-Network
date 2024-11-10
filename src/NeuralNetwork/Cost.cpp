#include "Cost.h"

namespace Cost {
	CostType GetFunctionEnum(double (*costFunc)(Vector, Vector)) {
		if (costFunc == MeanSquaredError)
			return CostType::MeanSquaredErrorCost;
		else if (costFunc == BinaryCrossEntropy)
			return CostType::BinaryCrossEntropyCost;
		else if (costFunc == CategoricalCrossEntropy)
			return CostType::CategoricalCrossEntropyCost;
	}
	cfptr GetFunctionFromEnum(CostType funcType) {
		switch (funcType) {
		case CostType::MeanSquaredErrorCost:
			return MeanSquaredError;
			break;
		case CostType::BinaryCrossEntropyCost:
			return BinaryCrossEntropy;
			break;
		case CostType::CategoricalCrossEntropyCost:
			return CategoricalCrossEntropy;
			break;
		}
	}
	dfptr GetDerivativeFromEnum(CostType funcType) {
		switch (funcType) {
		case CostType::MeanSquaredErrorCost:
			return MeanSquaredErrorDerivative;
			break;
		case CostType::BinaryCrossEntropyCost:
			return BinaryCrossEntropyDerivative;
			break;
		case CostType::CategoricalCrossEntropyCost:
			return CategoricalCrossEntropyDerivative;
			break;
		}
	}

	double MeanSquaredError(Vector actualOutput, Vector expectedOutput) {
		return (actualOutput - expectedOutput).MagnitudeSqr() * 0.5;
	}
	double BinaryCrossEntropy(Vector actualOutput, Vector expectedOutput) {
		double cost = 0;
		for (int i = 0; i < actualOutput.size; i++) {
			cost += -expectedOutput(i) * log(actualOutput(i)) - (1 - expectedOutput(i)) * log(1 - actualOutput(i));
		}
		return cost;
	}
	double CategoricalCrossEntropy(Vector actualOutput, Vector expectedOutput) {
		double cost = 0;
		for (int i = 0; i < actualOutput.size; i++) {
			cost += -expectedOutput(i) * log(actualOutput(i));
		}
		return cost;
	}

	Vector MeanSquaredErrorDerivative(Vector actualOutput, Vector expectedOutput) {
		return actualOutput - expectedOutput;
	}
	Vector BinaryCrossEntropyDerivative(Vector actualOutput, Vector expectedOutput) {
		Vector output(actualOutput.size);
		for (int i = 0; i < output.size; i++) {
			output(i) = (actualOutput(i) - expectedOutput(i)) / (actualOutput(i) * (1 - actualOutput(i)));
		}
		return output;
	}
	Vector CategoricalCrossEntropyDerivative(Vector actualOutput, Vector expectedOutput) {
		Vector output(actualOutput.size);
		for (int i = 0; i < output.size; i++) {
			output(i) = -expectedOutput(i) / actualOutput(i);
		}
		return output;
	}
}