#ifndef AATGS_ADAM_H
#define AATGS_ADAM_H

/**
 * @file adam.h
 * @brief ADAM optimizer.
 */

#include "../utils/utils.h"
#include "../utils/memory.h"
#include "../utils/protos.h"
#include "optimizer.h"

typedef struct AATGS_OPTIMIZER_ADAM_STRUCT
{
   // Options
   int _maxits; // Maximum number of iterations
   AATGS_DOUBLE _tol; // Tolerance

   void *_problem; // problem

   int _n; // dimention of the problem
   
   AATGS_DOUBLE _beta1; // first moment
   AATGS_DOUBLE _beta2; // second moment
   AATGS_DOUBLE _epsilon; // small number to avoid division by zero
   AATGS_DOUBLE _alpha; // learning rate
   AATGS_DOUBLE *_m;
   AATGS_DOUBLE *_v;
   AATGS_DOUBLE *_m_hat;
   AATGS_DOUBLE *_v_hat;
   
   // History data
   int _nits; // Number of iterations
   AATGS_DOUBLE *_loss_history; // History of the loss function, the fist element is the initial loss
   int _keep_x_history; // Keep x history or not, since it can be large
   AATGS_DOUBLE *_x_history; // History of the points, the fist element is the initial point
   int _keep_grad_history; // Keep grad history or not, since it can be large
   AATGS_DOUBLE *_grad_history; // History of the gradient, the fist element is the initial gradient
   AATGS_DOUBLE *_grad_norm_history; // History of the gradient norm, the fist element is the initial gradient norm

}aatgs_optimizer_adam, *paatgs_optimizer_adam;

/**
 * @brief   Create an ADAM optimizer
 * @details Create an ADAM optimizer
 * @return  Pointer to the optimizer (void).
 */
void* AatgsOptimizationAdamCreate();

/**
 * @brief   Destroy an ADAM optimizer
 * @details Destroy an ADAM optimizer
 * @param   voptimizerp Pointer to the optimizer
 * @return  Returns 0 if successfull
 */
void AatgsOptimizationAdamFree( void **voptimizerp );

/**
 * @brief   Setup the options for an ADAM optimization
 * @details Setup the options for an ADAM optimization
 * @param[in]   voptimizer Pointer to the optimizer
 * @param[in]   maxits      Maximum number of iterations
 * @param[in]   tol         Tolerance
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationAdamSetOptions( void *voptimizer, 
                                       int maxits,
                                       AATGS_DOUBLE tol);

/**
 * @brief   Setup the parameters for an ADAM optimization
 * @details Setup the parameters for an ADAM optimization
 * @param[in]   voptimizer Pointer to the optimizer
 * @param[in]   beta1       First moment
 * @param[in]   beta2       Second moment
 * @param[in]   epsilon     Small number to avoid division by zero
 * @param[in]   alpha       Learning rate
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationAdamSetParameters( void *voptimizer,
                                          AATGS_DOUBLE beta1,
                                          AATGS_DOUBLE beta2,
                                          AATGS_DOUBLE epsilon,
                                          AATGS_DOUBLE alpha);

/**
 * @brief   Setup the history options for an ADAM optimization
 * @details Setup the history options for an ADAM optimization
 * @param[in]   voptimizer Pointer to the optimizer
 * @param[in]   keep_x_history          Keep x history
 * @param[in]   keep_grad_history       Keep grad history
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationAdamSetHistory( void *voptimizer,
                                       int keep_x_history,
                                       int keep_grad_history);

/**
 * @brief   Given a point x, proceed with the ADAM optimization
 * @details Given a point x, proceed with the ADAM optimization
 * @param[in]  voptimizer  Pointer to the optimizer
 * @param[in]  vproblem    Pointer to the problem
 * @param[in]  x           Point at which the loss function and its gradient are computed
 * @param[in]  x_final     Final point
 * @return  Returns 0 if maxits reached, 1 if tol reached, -1 if error occured
 */
int AatgsOptimizationAdamRun( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final);

/**
 * @brief   Get information about the ADAM optimization
 * @details Get information about the ADAM optimization
 * @param[in]     voptimizer  Pointer to the optimizer
 * @param[in,out] nitsp       Pointer to the number of iterations. Set to NULL if not needed.
 * @param[in,out] xp          Pointer to the point history. Set to NULL if not needed.
 * @param[in,out] lossp       Pointer to the loss function history. Set to NULL if not needed.
 * @param[in,out] gradp       Pointer to the gradient history. Set to NULL if not needed.
 * @param[in,out] grad_normp  Pointer to the gradient norm history. Set to NULL if not needed.
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationAdamGetInfo( void *voptimizer, int *nitsp, AATGS_DOUBLE **xp, AATGS_DOUBLE **lossp, AATGS_DOUBLE **gradp, AATGS_DOUBLE **grad_normp);

#endif