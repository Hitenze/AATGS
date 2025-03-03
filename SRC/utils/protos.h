#ifndef AATGS_PROTOS_H
#define AATGS_PROTOS_H

/**
 * @file protos.h
 * @brief Prototypes of functions, depends on utils.h for the definition of AATGS_USING_FLOAT32
 */

#include "utils.h"

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