#pragma once

#include <NeuralNetwork/Maths/Vector.h>

namespace Cost {
	enum CostType {
		MeanSquaredErrorCost,
		BinaryCrossEntropyCost,
		CategoricalCrossEntropyCost
	};
	
	class CostFunction {
	public:
		virtual double Evaluate(Vector actualOutput, Vector expectedOutput) = 0;
		virtual Vector EvaluateDerivative(Vector actualOutput, Vector expectedOutput) = 0;
		virtual double GetCostType() const = 0;
	};

	class MeanSquaredError : public CostFunction {
	public:
		double Evaluate(Vector actualOutput, Vector expectedOutput) override;
		Vector EvaluateDerivative(Vector actualOutput, Vector expectedOutput) override;
		double GetCostType() const override;
	};
	class BinaryCrossEntropy: public CostFunction {
	public:
		double Evaluate(Vector actualOutput, Vector expectedOutput) override;
		Vector EvaluateDerivative(Vector actualOutput, Vector expectedOutput) override;
		double GetCostType() const override;
	};
	class CategoricalCrossEntropy : public CostFunction {
	public:
		double Evaluate(Vector actualOutput, Vector expectedOutput) override;
		Vector EvaluateDerivative(Vector actualOutput, Vector expectedOutput) override;
		double GetCostType() const override;
	};

	CostFunction* GetFunctionFromEnum(CostType funcType);
}