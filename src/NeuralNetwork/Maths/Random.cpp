#include <NeuralNetwork/Maths/Random.h>

int RandomFromRange(int lowerLimit, int upperLimit) {
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> dist(lowerLimit, upperLimit); // distribution in range [lowerLimit, upperLimit]
	return dist(rng);
}