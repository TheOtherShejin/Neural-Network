#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <NeuralNetwork/NeuralNetwork.h>
#include <vector>

void LoadNormalizeAndSave(std::string path, std::string savePath);
std::vector<DataPoint> LoadIntoDataset(std::string path, float validationSplit = 0.0f, std::vector<DataPoint>* validation_dataset = nullptr);