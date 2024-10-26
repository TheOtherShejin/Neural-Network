#pragma once

#include <vector>
#include <iostream>

class Vector {
private:
	std::vector<double> elements;

	bool CheckForErrors(int otherSize);
public:
	int size;

	Vector();
	Vector(int size);
	Vector(std::vector<double> elements);

	Vector operator+(Vector other);
	Vector operator-(Vector other);
	Vector operator*(Vector other); // Hadamard Product
	Vector operator*(double other);

	void operator+=(Vector other);
	void operator-=(Vector other);
	void operator*=(Vector other); // Hadamard Product
	void operator*=(double other);
	double& operator()(int i);

	double Dot(Vector other);
	double Magnitude();
	void SetZero();
	void Print();
};