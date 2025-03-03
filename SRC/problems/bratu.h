#ifndef AATGS_BRATU_H
#define AATGS_BRATU_H

/**
 * @file bratu.h
 * @brief   Bratu problem
 * @details Bratu problem u_xx + u_yy + alpha * u_x + lambda * exp(u) = 0 \n
 *          Domain: [0, 1] x [0, 1] \n
 *          Dirichlet boundary conditions: u = 0 on the boundary
 */

#include "problem.h"

typedef struct AATGS_PROBLEM_BRATU_STRUCT
{
   AATGS_DOUBLE   _lh;
   int            _nx;
   int            _n;

   int            *_A_i;
   int            *_A_j;
   AATGS_DOUBLE   *_A_a;

   AATGS_DOUBLE   *_dwork;
}aatgs_problem_bratu, *paatgs_problem_bratu;

/**
 * @brief   Create the Bratu problem
 * @details Create the Bratu problem
 * @param   alpha   Parameter alpha
 * @param   lambda  Parameter lambda
 * @param   nx      Number of points in each direction
 * @return  Returns a pointer to the problem data structure
 */
void* AatgsProblemBratuCreate( AATGS_DOUBLE alpha, AATGS_DOUBLE lambda, int nx );

/**
 * @brief   Free the Bratu problem
 * @details Free the Bratu problem
 * @param   vproblemp   Pointer to the problem data structure
 */
void AatgsProblemBratuFree( void **vproblemp);

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
int AatgsProblemBratuLoss( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *lossp, AATGS_DOUBLE *dloss);

/**
 * @brief   Given a point x, compute the Hessian of the loss function
 * @details Given a point x, compute the Hessian of the loss function. \n 
 *          If hessp is NULL, the Hessian is not computed.
 * @param   vproblem    Pointer to the problem data structure
 * @param   x           Point at which the Hessian is computed
 * @param   hessp       Pointer to the Hessian of the loss function. If NULL, the Hessian is not computed. Must be preallocated.
 * @return  Returns 0 if successfull
 */
int AatgsProblemBratuHess( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *hess);

#endif