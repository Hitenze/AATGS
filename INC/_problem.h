#ifndef AATGS_PROBLEM_HEADER_H
#define AATGS_PROBLEM_HEADER_H
#include "_utils.h"
#include "_ops.h"

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
