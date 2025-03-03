#include "adam.h"
#include "../ops/vecops.h"

void* AatgsOptimizationAdamCreate()
{
   paatgs_optimizer optimizer = NULL;
   AATGS_MALLOC(optimizer, 1, aatgs_optimizer);
   paatgs_optimizer_adam adam = NULL;
   AATGS_MALLOC(adam, 1, aatgs_optimizer_adam);

   adam->_maxits = 1000;
   adam->_tol = 1e-6;

   adam->_problem = NULL;

   adam->_n = 0;

   adam->_beta1 = 0.9;
   adam->_beta2 = 0.999;
   adam->_epsilon = 1e-8;
   adam->_alpha = 0.001;
   adam->_m = NULL;
   adam->_v = NULL;
   adam->_m_hat = NULL;
   adam->_v_hat = NULL;

   adam->_nits = 0;
   adam->_loss_history = NULL;
   adam->_keep_x_history = 0;
   adam->_x_history = NULL;
   adam->_keep_grad_history = 0;
   adam->_grad_history = NULL;
   adam->_grad_norm_history = NULL;

   optimizer->_optimizer_data = adam;
   optimizer->_run_optimizer = &AatgsOptimizationAdamRun;
   optimizer->_optimizer_get_info = &AatgsOptimizationAdamGetInfo;
   optimizer->_free_optimizer = &AatgsOptimizationAdamFree;

   return (void*)optimizer;
}

void AatgsOptimizationAdamFree( void **voptimizerp)
{
   if(*voptimizerp)
   {
      paatgs_optimizer optimizer = (paatgs_optimizer)*voptimizerp;
      paatgs_optimizer_adam adam = (paatgs_optimizer_adam)optimizer->_optimizer_data;
      if(adam)
      {
         AATGS_FREE(adam->_m);
         AATGS_FREE(adam->_v);
         AATGS_FREE(adam->_m_hat);
         AATGS_FREE(adam->_v_hat);
         AATGS_FREE(adam->_loss_history);
         AATGS_FREE(adam->_x_history);
         AATGS_FREE(adam->_grad_history);
         AATGS_FREE(adam->_grad_norm_history);
      }

      AATGS_FREE(adam);
   }
   AATGS_FREE(*voptimizerp);
}

int AatgsOptimizationAdamSetOptions( void *voptimizer, 
                                       int maxits,
                                       AATGS_DOUBLE tol)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_adam adam = (paatgs_optimizer_adam)optimizer->_optimizer_data;

   if(adam->_nits > 0)
   {
      printf("Error: cannot set options after the first optimization step.\n");
      return -1;
   }

   adam->_maxits = maxits;
   adam->_tol = tol;

   return 0;
}

int AatgsOptimizationAdamSetParameters( void *voptimizer,
                                          AATGS_DOUBLE beta1,
                                          AATGS_DOUBLE beta2,
                                          AATGS_DOUBLE epsilon,
                                          AATGS_DOUBLE alpha)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_adam adam = (paatgs_optimizer_adam)optimizer->_optimizer_data;

   if(adam->_nits > 0)
   {
      printf("Error: cannot set parameters after the first optimization step.\n");
      return -1;
   }

   adam->_beta1 = beta1;
   adam->_beta2 = beta2;
   adam->_epsilon = epsilon;
   adam->_alpha = alpha;

   return 0;
}

int AatgsOptimizationAdamSetHistory( void *voptimizer,
                                       int keep_x_history,
                                       int keep_grad_history)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_adam adam = (paatgs_optimizer_adam)optimizer->_optimizer_data;

   if(adam->_nits > 0)
   {
      printf("Error: cannot set history options after the first optimization step.\n");
      return -1;
   }

   adam->_keep_x_history = keep_x_history;
   adam->_keep_grad_history = keep_grad_history;

   return 0;
}

int AatgsOptimizationAdamRun( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final)
{
   /* set problem */
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_adam adam = (paatgs_optimizer_adam)optimizer->_optimizer_data;
   paatgs_problem problem = (paatgs_problem)vproblem;
   adam->_problem = problem;
   adam->_n = problem->_n;

   if(adam->_nits > 0)
   {
      printf("Error: cannot rerun optimization.\n");
      return -1;
   }

   /* Initialize ADAM */

   AATGS_CALLOC(adam->_m, adam->_n, AATGS_DOUBLE);
   AATGS_CALLOC(adam->_v, adam->_n, AATGS_DOUBLE);
   AATGS_MALLOC(adam->_m_hat, adam->_n, AATGS_DOUBLE);
   AATGS_MALLOC(adam->_v_hat, adam->_n, AATGS_DOUBLE);
   AATGS_CALLOC(adam->_loss_history, adam->_maxits+1, AATGS_DOUBLE);
   if(adam->_keep_x_history)
   {
      AATGS_CALLOC(adam->_x_history, (size_t)(adam->_maxits+1)*adam->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(adam->_x_history, 2*adam->_n, AATGS_DOUBLE);
   }
   if(adam->_keep_grad_history)
   {
      AATGS_CALLOC(adam->_grad_history, (size_t)(adam->_maxits+1)*adam->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(adam->_grad_history, adam->_n, AATGS_DOUBLE);
   }
   AATGS_CALLOC(adam->_grad_norm_history, adam->_maxits+1, AATGS_DOUBLE);

   AATGS_DOUBLE *x1 = adam->_x_history;
   AATGS_DOUBLE *x2 = adam->_x_history + adam->_n;
   AATGS_DOUBLE *x3 = NULL;
   AATGS_DOUBLE *xgrad = adam->_grad_history;

   AATGS_MEMCPY(x1, x, adam->_n, AATGS_DOUBLE);

   printf("Iteration  | Loss            | Grad norm     \n");

   /* LOOP ADAM */
   
   int i, err = 0;
   AATGS_DOUBLE beta1_hat, beta2_hat;

   for(adam->_nits = 0 ; adam->_nits <= adam->_maxits ; adam->_nits++)
   {
         
      // compute the loss function and its gradient
      err = problem->_loss(vproblem, x1, adam->_loss_history+adam->_nits, xgrad);

      if(err != 0)
      {
         printf("Loss failed\n");
         AATGS_MEMCPY(x_final, x1, adam->_n, AATGS_DOUBLE);
         return -1;
      }

      // now loss is in loss history and grad is in grad history
      beta1_hat = 1.0 / (1.0 - pow(adam->_beta1, adam->_nits+1.0));
      beta2_hat = 1.0 / (1.0 - pow(adam->_beta2, adam->_nits+1.0));
#ifdef AATGS_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
         for(i = 0 ; i < adam->_n ; i ++)
         {
            adam->_m[i] = adam->_beta1*adam->_m[i] + (1-adam->_beta1)*xgrad[i];
            adam->_v[i] = adam->_beta2*adam->_v[i] + (1-adam->_beta2)*xgrad[i]*xgrad[i];
            adam->_m_hat[i] = adam->_m[i]*beta1_hat;
            adam->_v_hat[i] = adam->_v[i]*beta2_hat;
            x2[i] = x1[i] - adam->_alpha*adam->_m_hat[i]/(sqrt(adam->_v_hat[i])+adam->_epsilon);
         }
      }
      else
      {
#endif
         for(i = 0 ; i < adam->_n ; i ++)
         {
            adam->_m[i] = adam->_beta1*adam->_m[i] + (1-adam->_beta1)*xgrad[i];
            adam->_v[i] = adam->_beta2*adam->_v[i] + (1-adam->_beta2)*xgrad[i]*xgrad[i];
            adam->_m_hat[i] = adam->_m[i]*beta1_hat;
            adam->_v_hat[i] = adam->_v[i]*beta2_hat;
            x2[i] = x1[i] - adam->_alpha*adam->_m_hat[i]/(sqrt(adam->_v_hat[i])+adam->_epsilon);
         }
#ifdef AATGS_USING_OPENMP
      }
#endif

      // check the stopping criteria
      adam->_grad_norm_history[adam->_nits] = AatgsVecNorm2( xgrad, adam->_n);

      // print the information for this iteration
      // we have 3 columns: iteration | loss | grad_norm
      // each should be 15 characters long
      // with vertical bars

      printf("%10d | %15.8e | %15.8e\n", adam->_nits, adam->_loss_history[adam->_nits], adam->_grad_norm_history[adam->_nits]);

      if(adam->_grad_norm_history[adam->_nits] < adam->_tol)
      {
         printf("Iteration stopped at %d with norm %e\n", adam->_nits, adam->_grad_norm_history[adam->_nits]);
         AATGS_MEMCPY(x_final, x2, adam->_n, AATGS_DOUBLE);
         return 1; // tol reached
      }

      if(adam->_keep_x_history)
      {
         x1 += adam->_n;
         x2 += adam->_n;
      }
      else
      {
         x3 = x1;
         x1 = x2;
         x2 = x3;
      }

      if(adam->_keep_grad_history)
      {
         xgrad += adam->_n;
      }

   }// end of main loop

   printf("Maximum number of iterations reached with norm %e\n", adam->_grad_norm_history[adam->_nits-1]);
   AATGS_MEMCPY(x_final, x2, adam->_n, AATGS_DOUBLE);
   return 0; // maxits reached
}

int AatgsOptimizationAdamGetInfo( void *voptimizer, int *nitsp, AATGS_DOUBLE **xp, AATGS_DOUBLE **lossp, AATGS_DOUBLE **gradp, AATGS_DOUBLE **grad_normp)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_adam adam = (paatgs_optimizer_adam)optimizer->_optimizer_data;

   if(nitsp)
   {
      *nitsp = adam->_nits;
   }

   if(xp)
   {
      *xp = adam->_x_history;
   }

   if(lossp)
   {
      *lossp = adam->_loss_history;
   }

   if(gradp)
   {
      *gradp = adam->_grad_history;
   }

   if(grad_normp)
   {
      *grad_normp = adam->_grad_norm_history;
   }

   return 0;
}
