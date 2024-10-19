#include "Vector.h"

Vector::Vector(int size) : size(size) {
	elements = std::vector<double>(size, 0);
}

Vector::Vector(std::vector<double> elements) 
	: elements(elements), size(elements.size()) {}

double& Vector::operator()(int i) {
	return elements[i];
}

double Vector::Dot(Vector& other) {
	if (other.size != size) {
		std::cerr << "Both vectors are of different dimension.\n";
		return 0;
	}

	double dotProduct = 0;
	for (int i = 0; i < size; i++) {
		dotProduct += elements[i] * other(i);
	}
	return dotProduct;
}
double Vector::Magnitude() {
	double magnitude = 0;
	for (auto& element : elements) {
		magnitude += element * element;
	}
	return sqrt(magnitude);
}
void Vector::Print() {
	std::cout << '(';
	for (int i = 0; i < elements.size()-1; i++) {
		std::cout << elements[i] << ", ";
	}
	std::cout << elements[size-1] << ")\n";
}