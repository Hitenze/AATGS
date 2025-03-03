#ifndef AATGS_UTIL_H
#define AATGS_UTIL_H

/**
 * @file util.h
 * @brief Basic functions, and defines, should depend on no other files
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
//#include <time.h>

#include <sys/stat.h>

// Some rountines are not good with too many threads
// change to #define AATGS_OPENMP_REDUCED_THREADS nthreads to use all threads for those rountines
#define AATGS_OPENMP_REDUCED_THREADS 1

#define AATGS_ARRAY_DEFAULT_SIZE 32
#define AARGS_ARRAY_EXPAND_FACTOR 1.3

// Double or Float
#ifdef AATGS_USING_FLOAT32
#define AATGS_DOUBLE float
#define AATGS_EPS FLT_EPSILON
#else
#define AATGS_DOUBLE double
#define AATGS_EPS DBL_EPSILON
#endif

#ifdef AATGS_USING_OPENMP
#include "omp.h"
#ifndef AATGS_DEFAULT_OPENMP_SCHEDULE
#define AATGS_DEFAULT_OPENMP_SCHEDULE schedule(static)
#endif
#endif

#define AATGS_MIN(a, b, c) {\
   (c) = (a) <= (b) ? (a) : (b);\
}

#define AATGS_MAX(a, b, c) {\
   (c) = (a) >= (b) ? (a) : (b);\
}

#define AATGS_SIGN(a, b) {\
   (b) = (a) > 0 ? 1 : (a) < 0 ? -1 : 0;\
}

/**
 * @brief   Get time in second. Always in double precision.
 * @details Get time in second. Always in double precision.
 * @return           Return time in second.
 */
double AatgsWtime();

#define AATGS_IO_MAX_STRING_LENGTH 1024

typedef enum AATGS_IO_TYPE_ENUM
{
   AATGS_IO_TYPE_INT = 0,
   AATGS_IO_TYPE_DOUBLE
}aatgs_io_type;

typedef struct AATGS_IO_HANDLE_STRUCT
{
   int _nargs;
   int _maxargs;
   char **_arg_abbrev;
   char **_arg_full;
   char **_arg_help;
   aatgs_io_type *_arg_type;
   void **_arg;
}aatgs_io_handle, *paatgs_io_handle;

void *AatgsIoHandleCreate();

void AatgsIoHandleFree(void **viohandle);

void AatgsIoHandleAddArg(void *viohandle, char *arg_abbrev, char *arg_full, char *arg_help, aatgs_io_type arg_type, void *arg);

void AatgsIoHandlePrintHelp(void *viohandle);

void AatgsIoHandlePringInfo(void *viohandle);

void AatgsIoHandlePhaseArgs(void *viohandle, int argc, char **argv);

/**
 * @brief   Create uniformly distributed points scaled by n^(1/d). Column major. Each column is a dimension.
 * @details Create uniformly distributed points scaled by n^(1/d). Column major. Each column is a dimension.
 * @param [in]       n: number of points
 * @param [in]       d: dimension of points
 * @return           Return the data matrix (d by n).
 */
void* AatgsMatrixUniformRandom(int n, int d);

/**
 * @brief   Print a dense matrix to terminal.
 * @details Print a dense matrix to terminal.
 * @param [in]       matrix: dense matrix, m by n
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @param [in]       ldim: leading dimension of matrix
 * @return           Return the data matrix (d by n).
 */
void AatgsTestPrintMatrix(AATGS_DOUBLE *matrix, int m, int n, int ldim);

/**
 * @brief   Print a dense matrix to file at a lower precision for validation.
 * @details Print a dense matrix to file at a lower precision for validation.
 * @param [in]       file: file pointer
 * @param [in]       matrix: dense matrix, m by n
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @param [in]       ldim: leading dimension of matrix
 * @return           Return the data matrix (d by n).
 */
void AatgsTestPrintMatrixToFile(FILE *file, AATGS_DOUBLE *matrix, int m, int n, int ldim);

/**
 * @brief   Print a sparse matrix to terminal.
 * @details Print a sparse matrix to terminal.
 * @param [in]       A_i: row index of sparse matrix
 * @param [in]       A_j: column index of sparse matrix
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @return           Return the data matrix (d by n).
 */
void AatgsTestPrintCSRMatrixPattern(int *A_i, int *A_j, int m, int n);

/**
 * @brief   Print a sparse matrix to file at a lower precision for validation.
 * @details Print a sparse matrix to file at a lower precision for validation.
 * @param [in]       file: file pointer
 * @param [in]       A_i: row index of sparse matrix
 * @param [in]       A_j: column index of sparse matrix
 * @param [in]       A_a: value of sparse matrix
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @return           Return the data matrix (d by n).
 */
void AatgsTestPrintCSRMatrixToFile(FILE *file, int *A_i, int *A_j, AATGS_DOUBLE *A_a, int m, int n);

/**
 * @brief   Print a sparse matrix to terminal.
 * @details Print a sparse matrix to terminal.
 * @param [in]       A_i: row index of sparse matrix
 * @param [in]       A_j: column index of sparse matrix
 * @param [in]       m: number of rows in matrix
 * @param [in]       n: number of columns in matrix
 * @param [in]       A_a: value of sparse matrix
 * @return           Return the data matrix (d by n).
 */
void AatgsTestPrintCSRMatrixVal(int *A_i, int *A_j, int m, int n, AATGS_DOUBLE *A_a);

#endif