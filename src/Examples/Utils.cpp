#include "Utils.h"

StringTokens Tokenize(std::string text, char separator) {
	std::stringstream ss(text);
	StringTokens tokens;
	std::string token;
	while (std::getline(ss, token, ' ')) {
		tokens.push_back(token);
	}
	return tokens;
}

int MaxVectorIndex(Vector vector) {
	int highestIndex = 0;
	double highestValue = vector(0);
	for (int i = 0; i < 10; i++) {
		if (vector(i) > highestValue) {
			highestValue = vector(i);
			highestIndex = i;
		}
	}
	return highestIndex;
}