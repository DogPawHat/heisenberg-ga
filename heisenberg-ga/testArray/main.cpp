#include <iostream>

int main(){
	int testArray[3][2];
	
	testArray[0][0] = 0;
	testArray[0][1] = 0;
	testArray[1][0] = 1;
	testArray[1][1] = 0;
	testArray[2][0] = 0;
	testArray[2][1] = 0;

	std::cout << testArray[2];
	std::cin.get();
}