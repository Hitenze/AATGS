#ifndef AATGS_OPTIMIZER_H
#define AATGS_OPTIMIZER_H

/*
 * @file optimizer.h
 * @brief Optimization function
 */

#include "../utils/utils.h"
#include "../utils/memory.h"
#include "../utils/protos.h"
#include "../problems/problem.h"

/**
 * @brief   Given a point x, proceed with the actual optimization
 * @details Given a point x, proceed with the actual optimization
 * @param[in]  voptimizer  Pointer to the optimizer
 * @param[in]  problem     Pointer to the problem
 * @param[in]  x           Point at which the loss function and its gradient are computed
 * @param[in]  x_final     Final point
 * @return  Returns 0 if successfull, 1 if maxits reached, 2 if tol reached, -1 if error occured
 */
typedef int (*func_optimization)( void *voptimizer, void *problem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final);

/**
 * @brief   Get information about the optimization
 * @details Get information about the optimization
 * @param[in]     voptimizer  Pointer to the optimizer
 * @param[in,out] nitsp       Pointer to the number of iterations. Set to NULL if not needed.
 * @param[in,out] xp          Pointer to the point history. Set to NULL if not needed.
 * @param[in,out] lossp       Pointer to the loss function history. Set to NULL if not needed.
 * @param[in,out] gradp       Pointer to the gradient history. Set to NULL if not needed.
 * @param[in,out] grad_normp  Pointer to the gradient norm history. Set to NULL if not needed.
 * @return  Returns 0 if successfull
 */
typedef int (*func_optimization_get_info)( void *voptimizer, int *nitsp, AATGS_DOUBLE **xp, AATGS_DOUBLE **lossp, AATGS_DOUBLE **gradp, AATGS_DOUBLE **grad_normp);

/**
 * @brief   The optimizer structure.
 * @details The optimizer structure.
 */
typedef struct AATGS_OPTIMIZER_STRUCT
{
   void *_optimizer_data; // Pointer to the optimizer data
   func_optimization _run_optimizer; // Run the optimizer
   func_optimization_get_info _optimizer_get_info; // Get information about the optimization
   func_free _free_optimizer; // Free the optimizer
}aatgs_optimizer, *paatgs_optimizer;

/**
 * @brief   The optimization structure.
 * @details The optimization structure.
 */
typedef struct AATGS_OPTIMIZATION_STRUCT
{
   void* _optimizer; // Pointer to the optimizer
   void* _problem; // Pointer to the problem
}aatgs_optimization, *paatgs_optimization;

/**
 * @brief   Create an optimization structure with given optimizer and problem.
 * @details Create an optimization structure with given optimizer and problem.
 * @param[in]  voptimizer  Pointer to the optimizer
 * @param[in]  vproblem    Pointer to the problem
 * @return  Returns 0 if successfull
 */
void* AatgsOptimizationCreate( void *voptimizer, void* vproblem);

/**
 * @brief   Run the optimization with given optimizer and problem.
 * @details Run the optimization with given optimizer and problem.
 * @param[in]  voptimization  Pointer to the optimization
 * @param[in]  x              Initial point
 * @param[in]  x_final        Final point
 * @return  Returns 0 if maxits reached, 1 if tol reached, -1 if error occured
 */
int AatgsOptimizationRun( void *voptimization, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final);

/**
 * @brief   Get information about the optimization
 * @details Get information about the optimization
 * @param[in]     voptimization  Pointer to the optimizer
 * @param[in,out] nitsp          Pointer to the number of iterations. Set to NULL if not needed.
 * @param[in,out] xp             Pointer to the point history. Set to NULL if not needed.
 * @param[in,out] lossp          Pointer to the loss function history. Set to NULL if not needed.
 * @param[in,out] gradp          Pointer to the gradient history. Set to NULL if not needed.
 * @param[in,out] grad_normp     Pointer to the gradient norm history. Set to NULL if not needed.
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationGetInfo( void *voptimization, int *nitsp, AATGS_DOUBLE **xp, AATGS_DOUBLE **lossp, AATGS_DOUBLE **gradp, AATGS_DOUBLE **grad_normp);

/**
 * @brief   Free an optimization structure only.
 * @details Free an optimization structure only.
 * @param[in]  voptimizationp Pointer to the optimization structure
 * @return  Returns 0 if successfull
 */
void AatgsOptimizationFree( void **voptimizationp);

/**
 * @brief   Free an optimization structure and the optimizer and problem.
 * @details Free an optimization structure and the optimizer and problem.
 * @param[in]  voptimizationp Pointer to the optimization structure
 * @return  Returns 0 if successfull
 */
void AatgsOptimizationFreeAll( void **voptimizationp);

#endif