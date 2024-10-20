#include "Matrix.h"

Matrix::Matrix() : rows(0), cols(0) {}
Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
	elements = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
}
Matrix::Matrix(std::vector<std::vector<double>> elements)
	: elements(elements), rows(elements.size()), cols(elements[0].size()) {}

double& Matrix::operator()(int row, int col) {
	return elements[row][col];
}

void Matrix::SetZero() {
	for (auto& row : elements) {
		std::fill(row.begin(), row.end(), 0);
	}
}
void Matrix::Print() {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			std::cout << elements[i][j] << ' ';
		}
		std::cout << '\n';
	}
}