#include <NeuralNetwork/Cost.h>

namespace Cost {
	double MeanSquaredError::Evaluate(Vector actualOutput, Vector expectedOutput) {
		return (actualOutput - expectedOutput).MagnitudeSqr() * 0.5;
	}
	Vector MeanSquaredError::EvaluateDerivative(Vector actualOutput, Vector expectedOutput) {
		return actualOutput - expectedOutput;
	}
	double MeanSquaredError::GetCostType() const {
		return Cost::MeanSquaredErrorCost;
	}

	double BinaryCrossEntropy::Evaluate(Vector actualOutput, Vector expectedOutput) {
		double cost = 0;
		for (int i = 0; i < actualOutput.size; i++) {
			cost += -expectedOutput(i) * log(actualOutput(i)) - (1 - expectedOutput(i)) * log(1 - actualOutput(i));
		}
		return cost;
	}
	Vector BinaryCrossEntropy::EvaluateDerivative(Vector actualOutput, Vector expectedOutput) {
		Vector output(actualOutput.size);
		for (int i = 0; i < output.size; i++) {
			output(i) = (actualOutput(i) - expectedOutput(i)) / (actualOutput(i) * (1 - actualOutput(i)));
		}
		return output;
	}
	double BinaryCrossEntropy::GetCostType() const {
		return Cost::BinaryCrossEntropyCost;
	}
	
	double CategoricalCrossEntropy::Evaluate(Vector actualOutput, Vector expectedOutput) {
		double cost = 0;
		for (int i = 0; i < actualOutput.size; i++) {
			cost += -expectedOutput(i) * log(actualOutput(i));
		}
		return cost;
	}
	Vector CategoricalCrossEntropy::EvaluateDerivative(Vector actualOutput, Vector expectedOutput) {
		Vector output(actualOutput.size);
		for (int i = 0; i < output.size; i++) {
			output(i) = -expectedOutput(i) / actualOutput(i);
		}
		return output;
	}
	double CategoricalCrossEntropy::GetCostType() const {
		return Cost::CategoricalCrossEntropyCost;
	}

	CostFunction* GetFunctionFromEnum(CostType funcType) {
		switch (funcType) {
		case CostType::MeanSquaredErrorCost:
			return new MeanSquaredError();
			break;
		case CostType::BinaryCrossEntropyCost:
			return new BinaryCrossEntropy();
			break;
		case CostType::CategoricalCrossEntropyCost:
			return new CategoricalCrossEntropy();
			break;
		}
	}
}