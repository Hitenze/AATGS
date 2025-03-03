#include "optimizer.h"

void* AatgsOptimizationCreate( void *voptimizer, void* vproblem)
{
   paatgs_optimization optimization = NULL;
   AATGS_MALLOC(optimization, 1, aatgs_optimization);

   optimization->_optimizer = voptimizer;
   optimization->_problem = vproblem;

   return (void*)optimization;
}

int AatgsOptimizationRun( void *voptimization, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final)
{
   paatgs_optimization optimization = (paatgs_optimization)voptimization;
   paatgs_optimizer optimizer = (paatgs_optimizer)optimization->_optimizer;
   return optimizer->_run_optimizer(optimization->_optimizer, optimization->_problem, x, x_final);
}

int AatgsOptimizationGetInfo( void *voptimization, int *nitsp, AATGS_DOUBLE **xp, AATGS_DOUBLE **lossp, AATGS_DOUBLE **gradp, AATGS_DOUBLE **grad_normp)
{
   paatgs_optimization optimization = (paatgs_optimization)voptimization;
   paatgs_optimizer optimizer = (paatgs_optimizer)optimization->_optimizer;
   return optimizer->_optimizer_get_info(optimization->_optimizer, nitsp, xp, lossp, gradp, grad_normp);
}

void AatgsOptimizationFree( void **voptimizationp)
{
   AATGS_FREE(*voptimizationp);
}

void AatgsOptimizationFreeAll( void **voptimizationp)
{
   if(*voptimizationp)
   {
      paatgs_optimization optimization = (paatgs_optimization)(*voptimizationp);
      paatgs_optimizer optimizer = (paatgs_optimizer)optimization->_optimizer;
      paatgs_problem problem = (paatgs_problem)optimization->_problem;
      optimizer->_free_optimizer((void**)&optimizer);
      problem->_free_problem((void**)&problem);
      AATGS_FREE(*voptimizationp);
   }
}
