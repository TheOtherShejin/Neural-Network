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