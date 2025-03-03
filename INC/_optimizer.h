#ifndef AATGS_OPTIMIZER_HEADER_H
#define AATGS_OPTIMIZER_HEADER_H
#include "_utils.h"
#include "_ops.h"
#include "_problem.h"

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

/**
 * @brief   Define enum of anderson acceleration algorithm type
 * @details Define enum of anderson acceleration algorithm type \n
 *          AATGS_OPTIMIZER_ANDERSON_TYPE_AATGS:      AATGS \n
 *          AATGS_OPTIMIZER_ANDERSON_TYPE_AARTGS:     AATGS reverse \n
 *          AATGS_OPTIMIZER_ANDERSON_TYPE_PINV:       FULL AA Pseudo inverse \n
 *          AATGS_OPTIMIZER_ANDERSON_TYPE_QR:         FULL AA QR \n
 */
typedef enum AATGS_OPTIMIZER_ANDERSON_TYPE_ENUM
{
   AATGS_OPTIMIZER_ANDERSON_TYPE_AATGS = 0,
   AATGS_OPTIMIZER_ANDERSON_TYPE_AARTGS,
   AATGS_OPTIMIZER_ANDERSON_TYPE_PINV,
   AATGS_OPTIMIZER_ANDERSON_TYPE_QR
}aatgs_optimizer_anderson_type;

/**
 * @brief   Define enum of anderson acceleration restart type
 * @details Define enum of anderson acceleration restart type \n
 *          AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NONE:         No restart \n
 *          AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NORM_BOUND:   Restart when the norm of the residual is too large \n
 *          AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_COOKTAIL:     Multiple restart constrains.
 */
typedef enum AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_ENUM
{
   AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NONE = 0,
   AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NORM_BOUND,
   AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_COOKTAIL
}aatgs_optimizer_anderson_restart_type;

typedef struct AATGS_OPTIMIZER_ANDERSON_STRUCT
{
   // Options
   int _maxits; // Maximum number of iterations
   AATGS_DOUBLE _tol; // Tolerance
   aatgs_optimizer_anderson_type _type; // Type of Anderson to use
   aatgs_optimizer_anderson_restart_type _restart_type; // Type of restart to use

   void *_problem; // problem

   int _n; // dimention of the problem
   
   int _wsize; // window size
   int _restart; // mandatory restart
   AATGS_DOUBLE _lr; // learning rate for the fixed point iteration
   AATGS_DOUBLE _beta; // mixing parameter
   AATGS_DOUBLE _safeguard; // restart tol if used
   
   // History data
   int _nits; // Number of iterations
   AATGS_DOUBLE *_loss_history; // History of the loss function, the fist element is the initial loss
   int _keep_x_history; // Keep x history or not, since it can be large
   AATGS_DOUBLE *_x_history; // History of the points, the fist element is the initial point
   int _keep_grad_history; // Keep grad history or not, since it can be large
   AATGS_DOUBLE *_grad_history; // History of the gradient, the fist element is the initial gradient
   AATGS_DOUBLE *_grad_norm_history; // History of the gradient norm, the fist element is the initial gradient norm

}aatgs_optimizer_anderson, *paatgs_optimizer_anderson;

/**
 * @brief   Create an Anderson acceleration optimizer
 * @details Create an Anderson acceleration optimizer
 * @return  Pointer to the optimizer (void).
 */
void* AatgsOptimizationAndersonCreate();

/**
 * @brief   Destroy an Anderson optimization
 * @details Destroy an Anderson optimization
 * @param   voptimizerp Pointer to the optimizer
 * @return  Returns 0 if successfull
 */
void AatgsOptimizationAndersonFree( void **voptimizerp );

/**
 * @brief   Setup the options for an Anderson optimization
 * @details Setup the options for an Anderson optimization
 * @param[in]   voptimizer Pointer to the optimizer
 * @param[in]   maxits      Maximum number of iterations
 * @param[in]   tol         Tolerance
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationAndersonSetOptions( void *voptimizer, 
                                       int maxits,
                                       AATGS_DOUBLE tol);

/**
 * @brief   Setup the parameters for an Anderson optimization
 * @details Setup the parameters for an Anderson optimization
 * @param[in]   voptimizer Pointer to the optimizer
 * @param[in]   lr          Learning rate
 * @param[in]   beta        Mixing parameter
 * @param[in]   wsize       Window size (negative means use full size)
 * @param[in]   restart     Restart dimension (negative means use full size)
 * @param[in]   safeguard   Restart tolerance
 * @param[in]   type        Type of Anderson to use
 * @param[in]   restart_type Type of restart to use
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationAndersonSetParameters( void *voptimizer,
                                       AATGS_DOUBLE lr,
                                       AATGS_DOUBLE beta,
                                       int wsize,
                                       int restart,
                                       AATGS_DOUBLE safeguard,
                                       aatgs_optimizer_anderson_type type,
                                       aatgs_optimizer_anderson_restart_type restart_type);

/**
 * @brief   Setup the history for an Anderson optimization
 * @details Setup the history for an Anderson optimization
 * @param[in]   voptimizer        Pointer to the optimizer
 * @param[in]   keep_x_history    Keep x history or not, since it can be large
 * @param[in]   keep_grad_history Keep grad history or not, since it can be large
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationAndersonSetHistory( void *voptimizer,
                                      int keep_x_history,
                                      int keep_grad_history);

/**
 * @brief   Given a point x, proceed with the Anderson optimization
 * @details Given a point x, proceed with the Anderson optimization
 * @param[in]  voptimizer  Pointer to the optimizer
 * @param[in]  vproblem    Pointer to the problem
 * @param[in]  x           Point at which the loss function and its gradient are computed
 * @param[in]  x_final     Final point
 * @return  Returns 0 if maxits reached, 1 if tol reached, -1 if error occured
 */
int AatgsOptimizationAndersonRun( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final);

/**
 * @brief   Get information about the Anderson optimization
 * @details Get information about the Anderson optimization
 * @param[in]     voptimizer  Pointer to the optimizer
 * @param[in,out] nitsp       Pointer to the number of iterations. Set to NULL if not needed.
 * @param[in,out] xp          Pointer to the point history. Set to NULL if not needed.
 * @param[in,out] lossp       Pointer to the loss function history. Set to NULL if not needed.
 * @param[in,out] gradp       Pointer to the gradient history. Set to NULL if not needed.
 * @param[in,out] grad_normp  Pointer to the gradient norm history. Set to NULL if not needed.
 * @return  Returns 0 if successfull
 */
int AatgsOptimizationAndersonGetInfo( void *voptimizer, int *nitsp, AATGS_DOUBLE **xp, AATGS_DOUBLE **lossp, AATGS_DOUBLE **gradp, AATGS_DOUBLE **grad_normp);

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
