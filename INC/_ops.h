#ifndef AATGS_OPS_HEADER_H
#define AATGS_OPS_HEADER_H
#include "_utils.h"

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

/**
 * @brief   Compute the general matrix-vector product y = alpha*A*x + beta*y.
 * @details Compute the general matrix-vector product y = alpha*A*x + beta*y.
 * @param[in]   data    Pointer to the matrix.
 * @param[in]   m       Number of rows of the matrix.
 * @param[in]   n       Number of columns of the matrix.
 * @param[in]   alpha   Scaling factor for the matrix.
 * @param[in]   x       Pointer to the vector.
 * @param[in]   beta    Scaling factor for the y vector.
 * @param[in,out]   y   Pointer to the vector.
 * @return  0 if success.
 */
int AatgsDenseMatGemv(void *data, char trans, int m, int n, AATGS_DOUBLE alpha, AATGS_DOUBLE *x, AATGS_DOUBLE beta, AATGS_DOUBLE *y);

/**
 * @brief   Modified Gram-Schmidt orthogonalization for Arnoldi.
 * @details Modified Gram-Schmidt orthogonalization for Arnoldi.
 * @param[in,out] w           Pointer to the vector.
 * @param[in]     n           Dimension of the vector.
 * @param[in]     kdim        Dimension of the Krylov subspace.
 * @param[in]     V           Pointer to the matrix of the Krylov subspace.
 * @param[in]     H           Pointer to the Hessberg matrix.
 * @param[in]     t           Vector norm of the crruent vector.
 * @param[in]     k           Current iteration.
 * @param[in]     tol_orth    Tolerance for the orthogonality.
 * @param[in]     tol_reorth  Tolerance for the reorthogonalization.
 * @return  0 if success.
 */
int AatgsModifiedGS( AATGS_DOUBLE *w, int n, int kdim, AATGS_DOUBLE *V, AATGS_DOUBLE *H, AATGS_DOUBLE *t, int k, AATGS_DOUBLE tol_orth, AATGS_DOUBLE tol_reorth);

/**
 * @brief   Compute the matrix-vector product y = alpha*A*x + beta*y of CSR matrix.
 * @details Compute the matrix-vector product y = alpha*A*x + beta*y of CSR matrix.
 * @param[in]   ia      Pointer to the row pointer.
 * @param[in]   ja      Pointer to the column indices.
 * @param[in]   aa      Pointer to the non-zero values.
 * @param[in]   nrows   Number of rows of the matrix.
 * @param[in]   ncols   Number of columns of the matrix.
 * @param[in]   trans   Transpose flag.
 * @param[in]   alpha   Scaling factor for the matrix.
 * @param[in]   x       Pointer to the x vector.
 * @param[in]   beta    Scaling factor for the y vector.
 * @param[in,out]   y   Pointer to the y vector.
 * @return  0 if success.
 */
int AatgsCsrMv( int *ia, int *ja, AATGS_DOUBLE *aa, int nrows, int ncols, char trans, AATGS_DOUBLE alpha, AATGS_DOUBLE *x, AATGS_DOUBLE beta, AATGS_DOUBLE *y);
#endif
