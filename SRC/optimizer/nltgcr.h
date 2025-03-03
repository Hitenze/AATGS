#ifndef AATGS_NLTGCR_H
#define AATGS_NLTGCR_H

/**
 * @file nltgcr.h
 * @brief NLTGCR optimizer
 */

#include "../utils/utils.h"
#include "../utils/memory.h"
#include "../utils/protos.h"
#include "optimizer.h"

/**
 * @brief   Define enum of NLTGCR algorithm type
 * @details Define enum of NLTGCR algorithm type \n
 *          AATGS_OPTIMIZER_NLTGCR_TYPE_LINEAR:       Linear version of NLTGCR \n
 *          AATGS_OPTIMIZER_NLTGCR_TYPE_NONLINEAR:    Nonlinear version \n
 *          AATGS_OPTIMIZER_NLTGCR_TYPE_ADAPTIVE:     Adaptive version \n
 *          AATGS_OPTIMIZER_NLTGCR_TYPE_HESS_PRECOND: Preconditioner enabled version
 */
typedef enum AATGS_OPTIMIZER_NLTGCR_TYPE_ENUM
{
   AATGS_OPTIMIZER_NLTGCR_TYPE_LINEAR = 0,
   AATGS_OPTIMIZER_NLTGCR_TYPE_NONLINEAR,
   AATGS_OPTIMIZER_NLTGCR_TYPE_ADAPTIVE,
   AATGS_OPTIMIZER_NLTGCR_TYPE_HESS_PRECOND
}aatgs_optimizer_nltgcr_type;

/**
 * @brief   Define enum of NLTGCR restart type
 * @details Define enum of NLTGCR restart type \n
 *          AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NONE:          No restart \n
 *          AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NORM_BOUND:    Restart when the norm of the residual is too large
 */
typedef enum AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_ENUM
{
   AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NONE = 0,
   AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NORM_BOUND
}aatgs_optimizer_nltgcr_restart_type;

typedef struct AATGS_OPTIMIZER_NLTGCR_STRUCT
{
   // Options
   int _maxits; // Maximum number of iterations
   AATGS_DOUBLE _tol; // Tolerance
   aatgs_optimizer_nltgcr_type _type; // Type of Anderson to use
   aatgs_optimizer_nltgcr_restart_type _restart_type; // Type of restart to use

   void *_problem; // problem

   int _n; // dimention of the problem
   
   int _max_ls; // max line search steps
   AATGS_DOUBLE _agc; // Armijo-Goldstein condition
   int _wsize; // window size
   int _restart; // mandatory restart
   AATGS_DOUBLE _lr; // learning rate for the fixed point iteration
   AATGS_DOUBLE _safeguard; // restart tol if used
   int _complex_diff; // using complex diff?
   
   // History data
   int _nits; // Number of iterations
   AATGS_DOUBLE *_loss_history; // History of the loss function, the fist element is the initial loss
   int _keep_x_history; // Keep x history or not, since it can be large
   AATGS_DOUBLE *_x_history; // History of the points, the fist element is the initial point
   int _keep_grad_history; // Keep grad history or not, since it can be large
   AATGS_DOUBLE *_grad_history; // History of the gradient, the fist element is the initial gradient
   AATGS_DOUBLE *_grad_norm_history; // History of the gradient norm, the fist element is the initial gradient norm

}aatgs_optimizer_nltgcr, *paatgs_optimizer_nltgcr;

/**
 * @brief   Create a NLTGCR optimizer
 * @details Create a NLTGCR optimizer
 * @return  Pointer to the optimizer (void).
 */
void* AatgsOptimizationNltgcrCreate();

/**
 * @brief   Destroy a NLTGCR optimizer
 * @details Destroy a NLTGCR optimizer
 * @param   voptimizerp Pointer to the optimizer
 * @return  Returns 0 if successfull
 */
void AatgsOptimizationNltgcrFree( void **voptimizerp );

/**
 * @brief   Setup the options for a NLTGCR optimizer
 * @details Setup the options for a NLTGCR optimizer
 * @param[in]   voptimizer Pointer to the optimizer
 * @param[in]   maxits      Maximum number of iterations
 * @param[in]   tol         Tolerance
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationNltgcrSetOptions( void *voptimizer, 
                                       int maxits,
                                       AATGS_DOUBLE tol);

/**
 * @brief   Setup the parameters for a NLTGCR optimizer
 * @details Setup the parameters for a NLTGCR optimizer
 * @param[in]   voptimizer Pointer to the optimizer
 * @param[in]   lr            Learning rate
 * @param[in]   max_ls        Max line search steps
 * @param[in]   wsize         Window size (negative means use full size)
 * @param[in]   agc           Armijo-Goldstein condition
 * @param[in]   restart       Restart dimension (negative means use full size)
 * @param[in]   complex_diff  Use complex diff for hessian? TODO: TO BE IMPLEMENTED
 * @param[in]   safeguard     Restart tol if used
 * @param[in]   type          Type of Anderson to use
 * @param[in]   restart_type Type of restart to use
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationNltgcrSetParameters( void *voptimizer,
                                       AATGS_DOUBLE lr,
                                       int max_ls,
                                       int wsize,
                                       AATGS_DOUBLE agc,
                                       int complex_diff,
                                       int restart,
                                       AATGS_DOUBLE safeguard,
                                       aatgs_optimizer_nltgcr_type type,
                                       aatgs_optimizer_nltgcr_restart_type restart_type);

/**
 * @brief   Setup the history for a NLTGCR optimizer
 * @details Setup the history for a NLTGCR optimizer
 * @param[in]   voptimizer        Pointer to the optimizer
 * @param[in]   keep_x_history    Keep x history or not, since it can be large
 * @param[in]   keep_grad_history Keep grad history or not, since it can be large
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationNltgcrSetHistory( void *voptimizer,
                                      int keep_x_history,
                                      int keep_grad_history);

/**
 * @brief   Given a point x, proceed with the NLTGCR optimization
 * @details Given a point x, proceed with the NLTGCR optimization
 * @param[in]  voptimizer  Pointer to the optimizer
 * @param[in]  vproblem    Pointer to the problem
 * @param[in]  x           Point at which the loss function and its gradient are computed
 * @param[in]  x_final     Final point
 * @return  Returns 0 if maxits reached, 1 if tol reached, -1 if error occured
 */
int AatgsOptimizationNltgcrRun( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final);

/**
 * @brief   Get information about the NLTGCR optimization
 * @details Get information about the NLTGCR optimization
 * @param[in]     voptimizer  Pointer to the optimizer
 * @param[in,out] nitsp       Pointer to the number of iterations. Set to NULL if not needed.
 * @param[in,out] xp          Pointer to the point history. Set to NULL if not needed.
 * @param[in,out] lossp       Pointer to the loss function history. Set to NULL if not needed.
 * @param[in,out] gradp       Pointer to the gradient history. Set to NULL if not needed.
 * @param[in,out] grad_normp  Pointer to the gradient norm history. Set to NULL if not needed.
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationNltgcrGetInfo( void *voptimizer, int *nitsp, AATGS_DOUBLE **xp, AATGS_DOUBLE **lossp, AATGS_DOUBLE **gradp, AATGS_DOUBLE **grad_normp);

#endif