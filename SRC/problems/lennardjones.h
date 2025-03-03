#ifndef AATGS_LENNARDJONES_H
#define AATGS_LENNARDJONES_H

/**
 * @file lennardjones.h
 * @brief   Lennard-Jones problem
 * @details Lennard-Jones problem, energy and forces for Lennard Jones potential
 */

#include "problem.h"

typedef struct AATGS_PROBLEM_LENNARDJONES_STRUCT
{
   int            _dim;
   int            _nat;
   int            _n;

   AATGS_DOUBLE   *_dwork;
}aatgs_problem_lennardjones, *paatgs_problem_lennardjones;

/**
 * @brief   Create the Lennard-Jones problem
 * @details Create the Lennard-Jones problem
 * @param   nat         Number of atoms.
 * @param   dim         Dimension of the space.
 * @return  Returns a pointer to the problem data structure
 */
void* AatgsProblemLennardJonesCreate( int nat, int dim);

/**
 * @brief   Free the Lennard-Jones problem
 * @details Free the Lennard-Jones problem
 * @param   vproblemp   Pointer to the problem data structure
 */
void AatgsProblemLennardJonesFree( void **vproblemp);

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
int AatgsProblemLennardJonesLoss( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *lossp, AATGS_DOUBLE *dloss);

/**
 * @brief   Given a point x, compute the Hessian of the loss function
 * @details Given a point x, compute the Hessian of the loss function. \n 
 *          If hessp is NULL, the Hessian is not computed.
 * @param   vproblem    Pointer to the problem data structure
 * @param   x           Point at which the Hessian is computed
 * @param   hessp       Pointer to the Hessian of the loss function. If NULL, the Hessian is not computed. Must be preallocated.
 * @return  Returns 0 if successfull
 */
int AatgsProblemLennardJonesHess( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *hess);

#endif