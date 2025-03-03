#ifndef AATGS_PROBLEM_H
#define AATGS_PROBLEM_H

/**
 * @file problems.h
 * @brief Optimization problems, all problems should follow this interface
 */

#include "../utils/utils.h"
#include "../utils/memory.h"
#include "../utils/protos.h"

/**
 * @brief   Given a point x, compute the loss function and its gradient
 * @details Given a point x, compute the loss function and its gradient. \n 
 *          If lossp is NULL, the loss function is not computed. If dlossp is NULL, the gradient is not computed.
 * @param   vproblem    Pointer to the problem data structure. NOTE: The AATGS_PROBLEM_STRUCT.
 * @param   x           Point at which the loss function and its gradient are computed
 * @param   transform   Transform to be used
 * @param   lossp       Pointer to the loss function. If NULL, the loss function is not computed
 * @param   dloss       Pointer to the gradient of the loss function. If NULL, the gradient is not computed. Must be preallocated.
 * @return  Returns 0 if successfull
 */
typedef int (*func_loss)( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *lossp, AATGS_DOUBLE *dloss);

/**
 * @brief   Given a point x, compute the Hessian of the loss function
 * @details Given a point x, compute the Hessian of the loss function. \n 
 *          If hessp is NULL, the Hessian is not computed.
 * @param   vproblem    Pointer to the problem data structure. NOTE: The AATGS_PROBLEM_STRUCT.
 * @param   x           Point at which the Hessian is computed
 * @param   hessp       Pointer to the Hessian of the loss function. If NULL, the Hessian is not computed. Must be preallocated.
 * @return  Returns 0 if successfull
 */
typedef int (*func_hess)( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *hess);

typedef struct AATGS_PROBLEM_STRUCT
{
   void *_problem_data;
   int _n;
   func_loss _loss;
   func_hess _hess;
   func_free _free_problem;
}aatgs_problem, *paatgs_problem;

#endif