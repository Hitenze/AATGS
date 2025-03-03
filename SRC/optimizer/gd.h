#ifndef AATGS_GD_H
#define AATGS_GD_H

/**
 * @file gd.h
 * @brief GD iteration (gradient descent). Note that x = x - lr*grad is a fixed point iteration.
 */

#include "../utils/utils.h"
#include "../utils/memory.h"
#include "../utils/protos.h"
#include "optimizer.h"

typedef struct AATGS_OPTIMIZER_GD_STRUCT
{
   // Options
   int _maxits; // Maximum number of iterations
   AATGS_DOUBLE _tol; // Tolerance

   void *_problem; // problem

   int _n; // dimention of the problem
   
   AATGS_DOUBLE _lr; // learning rate
   
   // History data
   int _nits; // Number of iterations
   AATGS_DOUBLE *_loss_history; // History of the loss function, the fist element is the initial loss
   int _keep_x_history; // Keep x history or not, since it can be large
   AATGS_DOUBLE *_x_history; // History of the points, the fist element is the initial point
   int _keep_grad_history; // Keep grad history or not, since it can be large
   AATGS_DOUBLE *_grad_history; // History of the gradient, the fist element is the initial gradient
   AATGS_DOUBLE *_grad_norm_history; // History of the gradient norm, the fist element is the initial gradient norm

}aatgs_optimizer_gd, *paatgs_optimizer_gd;

/**
 * @brief   Create an GD optimizer
 * @details Create an GD optimizer
 * @return  Pointer to the optimizer (void).
 */
void* AatgsOptimizationGdCreate();

/**
 * @brief   Destroy an GD optimizer
 * @details Destroy an GD optimizer
 * @param   voptimizerp Pointer to the optimizer
 * @return  Returns 0 if successfull
 */
void AatgsOptimizationGdFree( void **voptimizerp );

/**
 * @brief   Setup the options for an GD optimization
 * @details Setup the options for an GD optimization
 * @param[in]   voptimizer Pointer to the optimizer
 * @param[in]   maxits      Maximum number of iterations
 * @param[in]   tol         Tolerance
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationGdSetOptions( void *voptimizer, 
                                       int maxits,
                                       AATGS_DOUBLE tol);

/**
 * @brief   Setup the parameters for an GD optimization
 * @details Setup the parameters for an GD optimization
 * @param[in]   voptimizer Pointer to the optimizer
 * @param[in]   lr          Learning rate
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationGdSetParameters( void *voptimizer,
                                       AATGS_DOUBLE lr);

/**
 * @brief   Setup the history for an GD optimization
 * @details Setup the history for an GD optimization
 * @param[in]   voptimizer        Pointer to the optimizer
 * @param[in]   keep_x_history    Keep x history or not, since it can be large
 * @param[in]   keep_grad_history Keep grad history or not, since it can be large
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationGdSetHistory( void *voptimizer,
                                      int keep_x_history,
                                      int keep_grad_history);

/**
 * @brief   Given a point x, proceed with the GD optimization
 * @details Given a point x, proceed with the GD optimization
 * @param[in]  voptimizer  Pointer to the optimizer
 * @param[in]  vproblem    Pointer to the problem
 * @param[in]  x           Point at which the loss function and its gradient are computed
 * @param[in]  x_final     Final point
 * @return  Returns 0 if maxits reached, 1 if tol reached, -1 if error occured
 */
int AatgsOptimizationGdRun( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final);

/**
 * @brief   Get information about the GD optimization
 * @details Get information about the GD optimization
 * @param[in]     voptimizer  Pointer to the optimizer
 * @param[in,out] nitsp       Pointer to the number of iterations. Set to NULL if not needed.
 * @param[in,out] xp          Pointer to the point history. Set to NULL if not needed.
 * @param[in,out] lossp       Pointer to the loss function history. Set to NULL if not needed.
 * @param[in,out] gradp       Pointer to the gradient history. Set to NULL if not needed.
 * @param[in,out] grad_normp  Pointer to the gradient norm history. Set to NULL if not needed.
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationGdGetInfo( void *voptimizer, int *nitsp, AATGS_DOUBLE **xp, AATGS_DOUBLE **lossp, AATGS_DOUBLE **gradp, AATGS_DOUBLE **grad_normp);

#endif