#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "../NeuralNetwork/Vector.h"

typedef std::vector<std::string> StringTokens;

StringTokens Tokenize(std::string text, char separator);
int MaxVectorIndex(Vector vector);