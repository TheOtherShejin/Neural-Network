#pragma once

class Application {
public:
	virtual void Run();
private:
	virtual void Init() = 0;
	virtual void Update() = 0;
};