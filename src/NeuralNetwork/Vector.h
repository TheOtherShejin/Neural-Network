#pragma once

#include <vector>
#include <iostream>

class Vector {
private:
	std::vector<double> elements;
public:
	int size;

	Vector(int size);
	Vector(std::vector<double> elements);

	double& operator()(int i);

	double Dot(Vector& other);
	double Magnitude();
	void Print();
};