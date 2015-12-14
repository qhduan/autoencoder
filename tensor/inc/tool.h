#ifndef TOOL_H
#define TOOL_H
/*!
 * \file blas.h
 * \brief contain some BLAS functions
 * 		asum: sum a vector
 * 		dot: dot two vector
 * 		nrm2: normal^2 (L2) of a vector
 * 		gemm: matrix product, C = a*A*B + b*C
 * \author qhduan.com
 * \date 2014-05-13
 */

namespace tensor {

void StartTimer (void);
float EndTimer (void);


}

#endif // TOOL_H
