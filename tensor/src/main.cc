
#include <iostream>
#include <fstream>

#include "../inc/tensor.hpp"

using namespace std;
using namespace tensor;

int main (void) {

	typedef Mat<GPU, float> mat;

	mat A = ones<mat>(3, 3);

	mat B = log(A);
	A.print("a");
	B.print("b");

	return 0;
}
