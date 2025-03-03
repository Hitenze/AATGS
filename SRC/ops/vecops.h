#ifndef AATGS_VECOPS_H
#define AATGS_VECOPS_H

/**
 * @file vecops.h
 * @brief Vector operations
 */

#include "../utils/utils.h"
#include "../utils/memory.h"
#include "../utils/protos.h"

/**
 * @brief   Compute the 2-norm of a vector.
 * @details Compute the 2-norm of a vector.
 * @param[in]   x   Pointer to the vector.
 * @param[in]   n   Length of the vector.
 * @return  The 2-norm of the vector.
 */
AATGS_DOUBLE AatgsVecNorm2(AATGS_DOUBLE *x, int n);

/**
 * @brief   Compute the dot product of two vectors.
 * @details Compute the dot product of two vectors.
 * @param[in]   x   Pointer to the first vector.
 * @param[in]   n   Length of the vectors.
 * @param[in]   y   Pointer to the second vector.
 * @return  The dot product of the two vectors.
 */
AATGS_DOUBLE AatgsVecDdot(AATGS_DOUBLE *x, int n, AATGS_DOUBLE *y);

/**
 * @brief   Compute the AXPY of two vectors. y = alpha*x + y.
 * @details Compute the AXPY of two vectors. y = alpha*x + y.
 * @param[in]   alpha   Scaling factor for the first vector.
 * @param[in]   x       Pointer to the first vector.
 * @param[in]   n       Length of the vectors.
 * @param[in]   y       Pointer to the second vector.
 */
void AatgsVecAxpy(AATGS_DOUBLE alpha, AATGS_DOUBLE *x, int n, AATGS_DOUBLE *y);

/**
 * @brief   Fill a vector with random values between 0 and 1.
 * @details Fill a vector with random values between 0 and 1.
 * @param[in,out]   x       Pointer to the vector.
 * @param[in]       n       Length of the vector.
 */
void AatgsVecRand(AATGS_DOUBLE *x, int n);

/**
 * @brief   Fill a vector with random values between -1 and 1.
 * @details Fill a vector with random values between -1 and 1.
 * @param[in,out]   x       Pointer to the vector.
 * @param[in]       n       Length of the vector.
 */
void AatgsVecRadamacher(AATGS_DOUBLE *x, int n);

/**
 * @brief   Fill a vector with a constant value.
 * @details Fill a vector with a constant value.
 * @param[in,out]   x       Pointer to the vector.
 * @param[in]       n       Length of the vector.
 * @param[in]       val     Value to be filled in the vector.
 */
void AatgsVecFill(AATGS_DOUBLE *x, int n, AATGS_DOUBLE val);

/**
 * @brief   Scale a vector.
 * @details Scale a vector.
 * @param[in,out]   x       Pointer to the vector.
 * @param[in]       n       Length of the vector.
 * @param[in]       scale   Scaling factor.
 */
void AatgsVecScale(AATGS_DOUBLE *x, int n, AATGS_DOUBLE scale);

/**
 * @brief   Fill a int vector with a constant value.
 * @details Fill a int vector with a constant value.
 * @param[in,out]   x       Pointer to the vector.
 * @param[in]       n       Length of the vector.
 * @param[in]       val     Value to be filled in the vector.
 */
void AatgsIVecFill(int *x, int n, int val);

#endif