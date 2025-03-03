#include "gd.h"
#include "../ops/vecops.h"

void* AatgsOptimizationGdCreate()
{
   paatgs_optimizer optimizer = NULL;
   AATGS_MALLOC(optimizer, 1, aatgs_optimizer);
   paatgs_optimizer_gd gd = NULL;
   AATGS_MALLOC(gd, 1, aatgs_optimizer_gd);

   gd->_maxits = 1000;
   gd->_tol = 1e-6;

   gd->_problem = NULL;

   gd->_n = 0;

   gd->_lr = 0.1;

   gd->_nits = 0;
   gd->_loss_history = NULL;
   gd->_keep_x_history = 0;
   gd->_x_history = NULL;
   gd->_keep_grad_history = 0;
   gd->_grad_history = NULL;
   gd->_grad_norm_history = NULL;

   optimizer->_optimizer_data = gd;
   optimizer->_run_optimizer = &AatgsOptimizationGdRun;
   optimizer->_optimizer_get_info = &AatgsOptimizationGdGetInfo;
   optimizer->_free_optimizer = &AatgsOptimizationGdFree;

   return (void*)optimizer;
}

void AatgsOptimizationGdFree( void **voptimizerp )
{
   if(*voptimizerp)
   {
      paatgs_optimizer optimizer = (paatgs_optimizer)*voptimizerp;
      paatgs_optimizer_gd gd = (paatgs_optimizer_gd)optimizer->_optimizer_data;
      if(gd)
      {
         AATGS_FREE(gd->_loss_history);
         AATGS_FREE(gd->_x_history);
         AATGS_FREE(gd->_grad_history);
         AATGS_FREE(gd->_grad_norm_history);
      }

      AATGS_FREE(gd);
   }
   AATGS_FREE(*voptimizerp);
}

int AatgsOptimizationGdSetOptions( void *voptimizer, 
                                    int maxits,
                                    AATGS_DOUBLE tol)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_gd gd = (paatgs_optimizer_gd)optimizer->_optimizer_data;

   if(gd->_nits > 0)
   {
      printf("Error: cannot set options after the first optimization step.\n");
      return -1;
   }

   gd->_maxits = maxits;
   gd->_tol = tol;

   return 0;
}

int AatgsOptimizationGdSetParameters( void *voptimizer,
                                          AATGS_DOUBLE lr)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_gd gd = (paatgs_optimizer_gd)optimizer->_optimizer_data;

   if(gd->_nits > 0)
   {
      printf("Error: cannot set parameters after the first optimization step.\n");
      return -1;
   }

   gd->_lr = lr;

   return 0;
}

int AatgsOptimizationGdSetHistory( void *voptimizer,
                                      int keep_x_history,
                                      int keep_grad_history)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_gd gd = (paatgs_optimizer_gd)optimizer->_optimizer_data;

   if(gd->_nits > 0)
   {
      printf("Error: cannot set history after the first optimization step.\n");
      return -1;
   }

   gd->_keep_x_history = keep_x_history;
   gd->_keep_grad_history = keep_grad_history;

   return 0;
}

int AatgsOptimizationGdRun( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final)
{
   /* set problem */
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_gd gd = (paatgs_optimizer_gd)optimizer->_optimizer_data;
   paatgs_problem problem = (paatgs_problem)vproblem;
   gd->_problem = problem;
   gd->_n = problem->_n;

   if(gd->_nits > 0)
   {
      printf("Error: cannot rerun optimization.\n");
      return -1;
   }

   /* Initialize GD */
   AATGS_CALLOC(gd->_loss_history, gd->_maxits+1, AATGS_DOUBLE);
   if(gd->_keep_x_history)
   {
      AATGS_CALLOC(gd->_x_history, (size_t)(gd->_maxits+1)*gd->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(gd->_x_history, 2*gd->_n, AATGS_DOUBLE);
   }
   if(gd->_keep_grad_history)
   {
      AATGS_CALLOC(gd->_grad_history, (size_t)(gd->_maxits+1)*gd->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(gd->_grad_history, gd->_n, AATGS_DOUBLE);
   }
   AATGS_CALLOC(gd->_grad_norm_history, gd->_maxits+1, AATGS_DOUBLE);

   AATGS_DOUBLE *x1 = gd->_x_history;
   AATGS_DOUBLE *x2 = gd->_x_history + gd->_n;
   AATGS_DOUBLE *x3 = NULL;
   AATGS_DOUBLE *xgrad = gd->_grad_history;

   AATGS_MEMCPY(x1, x, gd->_n, AATGS_DOUBLE);

   printf("Iteration  | Loss            | Grad norm     \n");

   /* LOOP GD */
   
   int i, err = 0;

   for(gd->_nits = 0 ; gd->_nits <= gd->_maxits ; gd->_nits++)
   {
         
      // compute the loss function and its gradient
      err = problem->_loss(vproblem, x1, gd->_loss_history+gd->_nits, xgrad);

      if(err != 0)
      {
         printf("Loss failed\n");
         AATGS_MEMCPY(x_final, x1, gd->_n, AATGS_DOUBLE);
         return -1;
      }

      // now loss is in loss history and grad is in grad history
 #ifdef AATGS_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
         for(i = 0 ; i < gd->_n ; i ++)
         {
            x2[i] = x1[i] - gd->_lr*xgrad[i];
         }
      }
      else
      {
#endif
         for(i = 0 ; i < gd->_n ; i ++)
         {
            x2[i] = x1[i] - gd->_lr*xgrad[i];
         }
#ifdef AATGS_USING_OPENMP
      }
#endif

      // check the stopping criteria
      gd->_grad_norm_history[gd->_nits] = AatgsVecNorm2( xgrad, gd->_n);

      // print the information for this iteration
      // we have 3 columns: iteration | loss | grad_norm
      // each should be 15 characters long
      // with vertical bars

      printf("%10d | %15.8e | %15.8e\n", gd->_nits, gd->_loss_history[gd->_nits], gd->_grad_norm_history[gd->_nits]);

      if(gd->_grad_norm_history[gd->_nits] < gd->_tol)
      {
         printf("Iteration stopped at %d with norm %e\n", gd->_nits, gd->_grad_norm_history[gd->_nits]);
         AATGS_MEMCPY(x_final, x2, gd->_n, AATGS_DOUBLE);
         return 1; // tol reached
      }

      if(gd->_keep_x_history)
      {
         x1 += gd->_n;
         x2 += gd->_n;
      }
      else
      {
         x3 = x1;
         x1 = x2;
         x2 = x3;
      }

      if(gd->_keep_grad_history)
      {
         xgrad += gd->_n;
      }
   }// end of main loop

   printf("Maximum number of iterations reached with norm %e\n", gd->_grad_norm_history[gd->_nits-1]);
   AATGS_MEMCPY(x_final, x2, gd->_n, AATGS_DOUBLE);
   return 0; // maxits reached
}

int AatgsOptimizationGdGetInfo( void *voptimizer, int *nitsp, AATGS_DOUBLE **xp, AATGS_DOUBLE **lossp, AATGS_DOUBLE **gradp, AATGS_DOUBLE **grad_normp)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_gd gd = (paatgs_optimizer_gd)optimizer->_optimizer_data;

   if(nitsp)
   {
      *nitsp = gd->_nits;
   }

   if(xp)
   {
      *xp = gd->_x_history;
   }

   if(lossp)
   {
      *lossp = gd->_loss_history;
   }

   if(gradp)
   {
      *gradp = gd->_grad_history;
   }

   if(grad_normp)
   {
      *grad_normp = gd->_grad_norm_history;
   }

   return 0;
}
