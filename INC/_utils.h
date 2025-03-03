#ifndef AATGS_UTILS_HEADER_H
#define AATGS_UTILS_HEADER_H

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

#define AATGS_MALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) AatgsMalloc( (size_t)(length)*sizeof(__VA_ARGS__));\
}

#define AATGS_CALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) AatgsCalloc( (size_t)(length)*sizeof(__VA_ARGS__), 1);\
}

#define AATGS_REALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) AatgsRealloc( (void*)(ptr), (size_t)(length)*sizeof(__VA_ARGS__));\
}

#define AATGS_MEMCPY(ptr_to, ptr_from, length, ...) {\
   AatgsMemcpy( (void*)(ptr_to), (void*)(ptr_from), (size_t)(length)*sizeof(__VA_ARGS__));\
}

#define AATGS_FREE( ptr) {\
   if(ptr){AatgsFreeHost( (void*)(ptr));}\
   (ptr) = NULL;\
}

/**
 * @brief Allocate memory on host
 * @details Allocate memory on host
 * @param[in] size Size of memory to be allocated
 * @return Pointer to allocated memory
 */
static inline void* AatgsMalloc(size_t size)
{
   void *ptr = NULL;
   ptr = malloc(size);
   return ptr;
}

/**
 * @brief Allocate memory on host and initialize to zero
 * @details Allocate memory on host and initialize to zero
 * @param[in] length Length of memory to be allocated
 * @param[in] unitsize Size of each unit of memory
 * @return Pointer to allocated memory
 */
static inline void* AatgsCalloc(size_t length, int unitsize)
{
   void *ptr = NULL;
   ptr = calloc(length, unitsize);
   return ptr;
}

/**
 * @brief Reallocate memory on host
 * @details Reallocate memory on host
 * @param[in,out] ptr Pointer to memory to be reallocated
 * @param[in] size Size of memory to be allocated
 * @return Pointer to allocated memory
 */
static inline void* AatgsRealloc(void *ptr, size_t size)
{
   return ptr ? realloc( ptr, size ) : malloc( size );
}

/**
 * @brief Copy memory on host
 * @details Copy memory on host
 * @param[in,out] ptr_to Pointer to memory to be copied to
 * @param[in] ptr_from Pointer to memory to be copied from
 * @param[in] size Size of memory to be copied
 */
static inline void AatgsMemcpy(void *ptr_to, void *ptr_from, size_t size)
{
#ifdef AATGS_USING_OPENMP
#ifndef AATGS_OPENMP_NO_MEMCPY
   // use openmp to copy if possible, might not gain on all systems
   if(!omp_in_parallel())
   {
      size_t i;
      #pragma omp parallel for AATGS_DEFAULT_OPENMP_SCHEDULE
      for(i = 0; i < size; i++)
      {
         ((char*)ptr_to)[i] = ((char*)ptr_from)[i];
      }
      return;
   }
#endif
#endif
   memcpy( ptr_to, ptr_from, size);
}

/**
 * @brief Free memory on host
 * @details Free memory on host
 * @param[in,out] ptr Pointer to memory to be freed
 */
static inline void AatgsFreeHost(void *ptr)
{
   free(ptr);
}

/* BLAS/LAPACK */

#ifdef AATGS_USING_FLOAT32

#ifdef AATGS_USING_MKL

#include "mkl.h"
#define AATGS_DSYSV       ssysv
#define AATGS_DSYEV       ssyev
#define AATGS_DAXPY       saxpy
#define AATGS_DDOT        sdot
#define AATGS_DGEMV       sgemv
#define AATGS_DSYMV       ssymv
#define AATGS_DPOTRF      spotrf
#define AATGS_TRTRS       strtrs
#define AATGS_DTRTRI      strtri
#define AATGS_DTRMM       strmm
#define AATGS_DTRSM       strsm
#define AATGS_DGESVD      sgesvd
#define AATGS_DGESVDX     sgesvdx
#define AATGS_DGEMM       sgemm
#define AATGS_DGESV       sgesv
#define AATGS_DGETRF      sgetrf
#define AATGS_DGETRI      sgetri
#define AATGS_DLANGE      slange
#define AATGS_DLANTB      slantb
#define AATGS_DLANSY      slansy
#define AATGS_DSTEV       sstev
#else
#define AATGS_DSYSV       ssysv_
#define AATGS_DSYEV       ssyev_
#define AATGS_DAXPY       saxpy_
#define AATGS_DDOT        sdot_
#define AATGS_DGEMV       sgemv_
#define AATGS_DSYMV       ssymv_
#define AATGS_DPOTRF      spotrf_
#define AATGS_TRTRS       strtrs_
#define AATGS_DTRTRI      strtri_
#define AATGS_DTRMM       strmm_
#define AATGS_DTRSM       strsm_
#define AATGS_DGESVD      sgesvd_
#define AATGS_DGESVDX     sgesvdx_
#define AATGS_DGEMM       sgemm_
#define AATGS_DGESV       sgesv_
#define AATGS_DGETRF      sgetrf_
#define AATGS_DGETRI      sgetri_
#define AATGS_DLANGE      slange_
#define AATGS_DLANTB      slantb_
#define AATGS_DLANSY      slansy_
#define AATGS_DSTEV       sstev_
#endif // #ifdef AATGS_USING_MKL

#ifdef AATGS_USING_ARPACK

#define AATGS_DSAUPD ssaupd_
#define AATGS_DSEUPD sseupd_

#endif

#else // #ifdef AATGS_USING_FLOAT32

#ifdef AATGS_USING_MKL

#include "mkl.h"
#define AATGS_DSYSV       dsysv
#define AATGS_DSYEV       dsyev
#define AATGS_DAXPY       daxpy
#define AATGS_DDOT        ddot
#define AATGS_DGEMV       dgemv
#define AATGS_DSYMV       dsymv
#define AATGS_DPOTRF      dpotrf
#define AATGS_TRTRS       dtrtrs
#define AATGS_DTRTRI      dtrtri
#define AATGS_DTRMM       dtrmm
#define AATGS_DTRSM       dtrsm
#define AATGS_DGESVD      dgesvd
#define AATGS_DGESVDX     dgesvdx
#define AATGS_DGEMM       dgemm
#define AATGS_DGESV       dgesv
#define AATGS_DGETRF      dgetrf
#define AATGS_DGETRI      dgetri
#define AATGS_DLANGE      dlange
#define AATGS_DLANTB      dlantb
#define AATGS_DLANSY      dlansy
#define AATGS_DSTEV       dstev
#else
#define AATGS_DSYSV       dsysv_
#define AATGS_DSYEV       dsyev_
#define AATGS_DAXPY       daxpy_
#define AATGS_DDOT        ddot_
#define AATGS_DGEMV       dgemv_
#define AATGS_DSYMV       dsymv_
#define AATGS_DPOTRF      dpotrf_
#define AATGS_TRTRS       dtrtrs_
#define AATGS_DTRTRI      dtrtri_
#define AATGS_DTRMM       dtrmm_
#define AATGS_DTRSM       dtrsm_
#define AATGS_DGESVD      dgesvd_
#define AATGS_DGESVDX     dgesvdx_
#define AATGS_DGEMM       dgemm_
#define AATGS_DGESV       dgesv_
#define AATGS_DGETRF      dgetrf_
#define AATGS_DGETRI      dgetri_
#define AATGS_DLANGE      dlange_
#define AATGS_DLANTB      dlantb_
#define AATGS_DLANSY      dlansy_
#define AATGS_DSTEV       dstev_
#endif // #ifdef AATGS_USING_MKL

#ifdef AATGS_USING_ARPACK

#define AATGS_DSAUPD dsaupd_
#define AATGS_DSEUPD dseupd_

#endif

#endif // #ifdef AATGS_USING_FLOAT32

#ifndef AATGS_USING_MKL

void AATGS_DSYSV(char *uplo, int *n, int *nrhs, AATGS_DOUBLE *a, int *lda, int *ipiv, AATGS_DOUBLE *b, int *ldb, AATGS_DOUBLE *work, int *lwork, int *info);

void AATGS_DSYEV(char *jobz, char *uplo, int *n, AATGS_DOUBLE *a, int *lda, AATGS_DOUBLE *w, AATGS_DOUBLE *work, int *lwork, int *info);

void AATGS_DAXPY(int *n, const AATGS_DOUBLE *alpha, const AATGS_DOUBLE *x, int *incx, AATGS_DOUBLE *y, int *incy);

AATGS_DOUBLE AATGS_DDOT(int *n, AATGS_DOUBLE *x, int *incx, AATGS_DOUBLE *y, int *incy);

void AATGS_DGEMV(char *trans, int *m, int *n, const AATGS_DOUBLE *alpha, const AATGS_DOUBLE *a,
               int *lda, const AATGS_DOUBLE *x, int *incx, const AATGS_DOUBLE *beta, AATGS_DOUBLE *y, int *incy);

void AATGS_DSYMV(char *uplo, int *n, AATGS_DOUBLE *alpha, AATGS_DOUBLE *a, int *lda, AATGS_DOUBLE *x, int *incx, AATGS_DOUBLE *beta, AATGS_DOUBLE *y, int *incy);

void AATGS_DPOTRF(char *uplo, int *n, AATGS_DOUBLE *a, int *lda, int *info);

void AATGS_TRTRS(char *uplo, char *trans, char *diag, int *n, int *nrhs, AATGS_DOUBLE *a, int *lda, AATGS_DOUBLE *b, int *ldb, int *info);

void AATGS_DTRTRI( char *uplo, char *diag, int *n, AATGS_DOUBLE *a, int *lda, int *info);

void AATGS_DTRMM( char *side, char *uplo, char *transa, char *diag, int *m, int *n, AATGS_DOUBLE *alpha, AATGS_DOUBLE *a, int *lda, AATGS_DOUBLE *b, int *ldb);

void AATGS_DTRSM( char *side, char *uplo, char *transa, char *diag, int *m, int *n, AATGS_DOUBLE *alpha, AATGS_DOUBLE *a, int *lda, AATGS_DOUBLE *b, int *ldb);

void AATGS_DGESVD(char *jobu, char *jobvt, int *m, int *n, AATGS_DOUBLE *a, int *lda, AATGS_DOUBLE *s, AATGS_DOUBLE *u, int *ldu,
                int *vt, int *ldvt, AATGS_DOUBLE *work, int *lwork, int *info);

void AATGS_DGESVDX( char *jobu, char *jobvt, char *range, int *m, int *n, AATGS_DOUBLE *a, int *lda, AATGS_DOUBLE *vl, AATGS_DOUBLE *vu,
                  int *il, int *iu, int *ns, AATGS_DOUBLE *s, AATGS_DOUBLE *u, int *ldu, int *vt, int *ldvt, AATGS_DOUBLE *work, int *lwork, int *info);

void AATGS_DGEMM(char *transa, char *transb, int *m, int *n, int *k, const AATGS_DOUBLE *alpha, const AATGS_DOUBLE *a, int *lda,
               const AATGS_DOUBLE *b, int *ldb, const AATGS_DOUBLE *beta, AATGS_DOUBLE *c, int *ldc);

void AATGS_DGESV( int *n, int *nrhs, AATGS_DOUBLE *a, int *lda, int *ipiv, AATGS_DOUBLE *b, int *ldb, int *info);

void AATGS_DGETRF( int *m, int *n, AATGS_DOUBLE *a, int *lda, int *ipiv, int *info);

void AATGS_DGETRI( int *n, AATGS_DOUBLE *a, int *lda, int *ipiv, AATGS_DOUBLE *work, int *lwork, int *info);

AATGS_DOUBLE AATGS_DLANGE( char *norm, int *m, int *n, AATGS_DOUBLE *A, int *lda, AATGS_DOUBLE *work);

AATGS_DOUBLE AATGS_DLANTB( char *norm, char *uplo, char *diag, int *n, int *k, AATGS_DOUBLE *A, int *lda, AATGS_DOUBLE *work);

AATGS_DOUBLE AATGS_DLANSY( char *norm, char *uplo, int *n, AATGS_DOUBLE *A, int *lda, AATGS_DOUBLE *work);

void AATGS_DSTEV( char *jobz, int *n, AATGS_DOUBLE *d, AATGS_DOUBLE *e, AATGS_DOUBLE *z, int *ldz, AATGS_DOUBLE *work, int *info);

#endif // #ifndef AATGS_USING_MKL

/* function prototypes */

/**
 * @brief   Create a data structure and set it to default values.
 * @details Create a data structure and set it to default values.
 * @return           Pointer to the data structure.
 */
typedef void* (*func_create)();

/**
 * @brief   Free a data structure.
 * @details This function is used to free a data structure.
 * @param [in,out]   str: pointer to the data structure to be freed.
 * @return           No return.
 */
typedef void (*func_free)(void **str);
#endif
