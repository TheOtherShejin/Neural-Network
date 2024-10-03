#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "NeuralNetwork.h"

void SaveModelToCSV(std::string path, NeuralNetwork* nn);
NeuralNetwork LoadModelFromCSV(std::string path);