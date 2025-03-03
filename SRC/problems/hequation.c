#include "hequation.h"
#include "../ops/vecops.h"
#include "../ops/matops.h"

void* AatgsProblemHeatEquationCreate( int n, AATGS_DOUBLE omega)
{
   paatgs_problem_hequation hequ = NULL;
   AATGS_MALLOC(hequ, 1, aatgs_problem_hequqtion);

   hequ->_n = n;
   hequ->_homegaN = 0.5*omega/(AATGS_DOUBLE)n;

   int i;
   AATGS_MALLOC(hequ->_mu, n, AATGS_DOUBLE);
   for(i = 0 ; i < n ; i ++)
   {
      hequ->_mu[i] = (i + 0.5)/(AATGS_DOUBLE)n;
   }

   size_t dwork_size = hequ->_n;
   AATGS_MALLOC(hequ->_dwork, dwork_size, AATGS_DOUBLE);

   paatgs_problem problem = NULL;
   AATGS_MALLOC(problem, 1, aatgs_problem);

   problem->_problem_data = (void*)hequ;
   problem->_n = hequ->_n;
   problem->_free_problem = &AatgsProblemHeatEquationFree;
   problem->_loss = &AatgsProblemHeatEquationLoss;
   problem->_hess = &AatgsProblemHeatEquationHess;
   
   return (void*)problem;
}

void AatgsProblemHeatEquationFree( void **vproblemp)
{
   if(*vproblemp)
   {
      paatgs_problem problem = (paatgs_problem)(*vproblemp);
      if(problem->_problem_data)
      {
         paatgs_problem_hequation hequ = (paatgs_problem_hequation)(problem->_problem_data);
         AATGS_FREE(hequ->_dwork);
         AATGS_FREE(hequ->_mu);
         AATGS_FREE(hequ);
      }
      AATGS_FREE(*vproblemp);
   }
}

int AatgsProblemHeatEquationLoss( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *lossp, AATGS_DOUBLE *dloss)
{
   paatgs_problem problem = (paatgs_problem)vproblem;
   paatgs_problem_hequation hequ = (paatgs_problem_hequation)(problem->_problem_data);

   AATGS_DOUBLE *g;
   if(dloss)
   {
      // output is needed
      g = dloss;
   }
   else
   {
      g = hequ->_dwork;
   }

   int i, j;
   
#ifdef AATGS_USING_OPENMP
   if(!omp_in_parallel() && omp_get_max_threads() > 1)
   {
      #pragma omp parallel for private(i, j) AATGS_DEFAULT_OPENMP_SCHEDULE
      for(i = 0 ; i < hequ->_n ; i ++)
      {
         g[i] = 0.0;
         for(j = 0 ; j < hequ->_n ; j ++)
         {
            g[i] += hequ->_mu[i]*x[j]/(hequ->_mu[i]+hequ->_mu[j]);
         }
         g[i] *= hequ->_homegaN;
         g[i] = x[i] - 1.0/(1.0 - g[i]);
      }
   }
   else
   {
#endif
      for(i = 0 ; i < hequ->_n ; i ++)
      {
         g[i] = 0.0;
         for(j = 0 ; j < hequ->_n ; j ++)
         {
            g[i] += hequ->_mu[i]*x[j]/(hequ->_mu[i]+hequ->_mu[j]);
         }
         g[i] *= hequ->_homegaN;
         g[i] = x[i] - 1.0/(1.0 - g[i]);
      }
#ifdef AATGS_USING_OPENMP
   }
#endif

   if(lossp)
   {
      // output is needed
      *lossp = AatgsVecNorm2(g, hequ->_n);
   }

   return 0;
}

int AatgsProblemHeatEquationHess( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *hess)
{
   printf("Error: Hessian not implemented yet.\n");
   return -1;
}
