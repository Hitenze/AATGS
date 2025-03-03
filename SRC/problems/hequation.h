#ifndef AATGS_HEQUATION_H
#define AATGS_HEQUATION_H

/**
 * @file hequation.h
 * @brief   H-equation problem
 * @details H-equation problem \n
 *          PDE: h_i = h_i = inv(1 - 0.5*(omega/N)*sum_j(mu_i*h_j/(mu_i+mu_j))) \n
 *              mu_i = (i - 0.5)/N \n
 *          omega typically 0.5, 0.99, or 1.0
 */

#include "problem.h"

typedef struct AATGS_PROBLEM_HEQUATION_STRUCT
{
   int            _n;
   AATGS_DOUBLE   _homegaN; // scaled omega, 0.5*omega/N
   AATGS_DOUBLE   *_mu;

   AATGS_DOUBLE   *_dwork;
}aatgs_problem_hequqtion, *paatgs_problem_hequation;

/**
 * @brief   Create the Heat Equation problem
 * @details Create the Heat Equation problem
 * @param   n           Size of problem.
 * @param   omega       Problem parameter.
 * @return  Returns a pointer to the problem data structure
 */
void* AatgsProblemHeatEquationCreate( int n, AATGS_DOUBLE omega);

/**
 * @brief   Free the Heat Equation problem
 * @details Free the Heat Equation problem
 * @param   vproblemp   Pointer to the problem data structure
 */
void AatgsProblemHeatEquationFree( void **vproblemp);

/**
 * @brief   Given a point x, compute the loss function and its gradient
 * @details Given a point x, compute the loss function and its gradient. \n 
 *          If lossp is NULL, the loss function is not computed. If dlossp is NULL, the gradient is not computed.
 * @param   vproblem    Pointer to the problem data structure
 * @param   x           Point at which the loss function and its gradient are computed
 * @param   transform   Transform to be used
 * @param   lossp       Pointer to the loss function. If NULL, the loss function is not computed
 * @param   dloss       Pointer to the gradient of the loss function. If NULL, the gradient is not computed. Must be preallocated.
 * @return  Returns 0 if successfull
 */
int AatgsProblemHeatEquationLoss( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *lossp, AATGS_DOUBLE *dloss);

/**
 * @brief   Given a point x, compute the Hessian of the loss function
 * @details Given a point x, compute the Hessian of the loss function. \n 
 *          If hessp is NULL, the Hessian is not computed.
 * @param   vproblem    Pointer to the problem data structure
 * @param   x           Point at which the Hessian is computed
 * @param   hessp       Pointer to the Hessian of the loss function. If NULL, the Hessian is not computed. Must be preallocated.
 * @return  Returns 0 if successfull
 */
int AatgsProblemHeatEquationHess( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *hess);

#endif