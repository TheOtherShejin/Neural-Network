#pragma once

#include <vector>
#include <iostream>

class Matrix {
private:
	std::vector<std::vector<double>> elements;
public:
	int rows, cols;

	Matrix();
	Matrix(int rows, int cols);
	Matrix(std::vector<std::vector<double>> elements);

	double& operator()(int row, int col);

	void SetZero();
	void Print();
};