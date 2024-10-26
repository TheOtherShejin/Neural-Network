#pragma once

#include <vector>
#include <iostream>

class Matrix {
private:
	std::vector<std::vector<double>> elements;

	bool CheckForErrors(int otherRows, int otherCols);
public:
	int rows, cols;

	Matrix();
	Matrix(int rows, int cols);
	Matrix(std::vector<std::vector<double>> elements);

	Matrix operator+(Matrix other);
	Matrix operator-(Matrix other);
	Matrix operator*(double other);

	void operator+=(Matrix other);
	void operator-=(Matrix other);
	void operator*=(double other);
	double& operator()(int row, int col);

	void SetZero();
	void Print();
};