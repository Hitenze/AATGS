#include "float.h"
#include "anderson.h"
#include "../ops/vecops.h"

void* AatgsOptimizationAndersonCreate()
{
   paatgs_optimizer optimizer = NULL;
   AATGS_MALLOC(optimizer, 1, aatgs_optimizer);
   paatgs_optimizer_anderson anderson = NULL;
   AATGS_MALLOC(anderson, 1, aatgs_optimizer_anderson);

   anderson->_maxits = 1000;
   anderson->_tol = 1e-6;
   anderson->_type = AATGS_OPTIMIZER_ANDERSON_TYPE_AATGS;
   anderson->_restart_type = AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NONE;

   anderson->_problem = NULL;

   anderson->_n = 0;

   anderson->_lr = 0.1;
   anderson->_beta = 0.1;
   anderson->_wsize = -1; // negative means use full size
   anderson->_restart = -1; // no mandatory restart
   anderson->_safeguard = 1e03;

   anderson->_nits = 0;
   anderson->_loss_history = NULL;
   anderson->_keep_x_history = 0;
   anderson->_x_history = NULL;
   anderson->_keep_grad_history = 0;
   anderson->_grad_history = NULL;
   anderson->_grad_norm_history = NULL;

   optimizer->_optimizer_data = anderson;
   optimizer->_run_optimizer = &AatgsOptimizationAndersonRun;
   optimizer->_optimizer_get_info = &AatgsOptimizationAndersonGetInfo;
   optimizer->_free_optimizer = &AatgsOptimizationAndersonFree;

   return (void*)optimizer;
}

void AatgsOptimizationAndersonFree( void **voptimizerp )
{
   if(*voptimizerp)
   {
      paatgs_optimizer optimizer = (paatgs_optimizer)*voptimizerp;
      paatgs_optimizer_anderson anderson = (paatgs_optimizer_anderson)optimizer->_optimizer_data;
      if(anderson)
      {
         AATGS_FREE(anderson->_loss_history);
         AATGS_FREE(anderson->_x_history);
         AATGS_FREE(anderson->_grad_history);
         AATGS_FREE(anderson->_grad_norm_history);
      }

      AATGS_FREE(anderson);
   }
   AATGS_FREE(*voptimizerp);
}

int AatgsOptimizationAndersonSetOptions( void *voptimizer, 
                                    int maxits,
                                    AATGS_DOUBLE tol)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_anderson anderson = (paatgs_optimizer_anderson)optimizer->_optimizer_data;

   if(anderson->_nits > 0)
   {
      printf("Error: cannot set options after the first optimization step.\n");
      return -1;
   }

   anderson->_maxits = maxits;
   anderson->_tol = tol;

   return 0;
}

int AatgsOptimizationAndersonSetParameters( void *voptimizer,
                                       AATGS_DOUBLE lr,
                                       AATGS_DOUBLE beta,
                                       int wsize,
                                       int restart,
                                       AATGS_DOUBLE safeguard,
                                       aatgs_optimizer_anderson_type type,
                                       aatgs_optimizer_anderson_restart_type restart_type)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_anderson anderson = (paatgs_optimizer_anderson)optimizer->_optimizer_data;

   if(anderson->_nits > 0)
   {
      printf("Error: cannot set parameters after the first optimization step.\n");
      return -1;
   }

   anderson->_lr = lr;
   anderson->_beta = beta;
   anderson->_wsize = wsize;
   anderson->_restart = restart;
   anderson->_safeguard = safeguard;
   anderson->_type = type;
   anderson->_restart_type = restart_type;

   return 0;
}

int AatgsOptimizationAndersonSetHistory( void *voptimizer,
                                      int keep_x_history,
                                      int keep_grad_history)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_anderson anderson = (paatgs_optimizer_anderson)optimizer->_optimizer_data;

   if(anderson->_nits > 0)
   {
      printf("Error: cannot set history after the first optimization step.\n");
      return -1;
   }

   anderson->_keep_x_history = keep_x_history;
   anderson->_keep_grad_history = keep_grad_history;

   return 0;
}

/**
 * @brief   Reverse Gram-Schmidt orthogonalization for Anderson.
 * @details Reverse Gram-Schmidt orthogonalization for Anderson.
 * @note    Helper function.
 * @param[in,out] U              Pointer to the U matrix.
 * @param[in,out] Q              Pointer to the Q matrix.
 * @param[in]     n              Dimension of the vector.
 * @param[in]     ncol           Number of columns of the two matrices.
 * @param[in]     cwsize         Current window size used.
 * @param[in]     idx            Index of the current vector.
 * @param[out]    f              Pointer to the f at current iteration.
 * @param[in,out] x_new          Pointer to the new x at current iteration.
 * @param[in]     x_old          Pointer to the old x at current iteration.
 * @param[in,out] theta          Pointer to the theta at current iteration.
 * @param[out]    t              Vector norm of the crruent vector after the orthogonalization.
 * @param[in]     beta           Mixing parameter.
 * @param[in]     restart_type   Restart type.
 * @param[in]     safeguard      Restart parameter.
 * @param[in,out] xrec           Pointer to the xrec at current iteration.
 * @param[in]     tol_orth       Tolerance for the orthogonality.
 * @param[in]     tol_reorth     Tolerance for the reorthogonalization.
 * @return  0 if success.
 */
int AatgsRGSAnderson( AATGS_DOUBLE *U, 
                              AATGS_DOUBLE *Q, 
                              int n, 
                              int ncol, 
                              int cwsize, 
                              int idx, 
                              AATGS_DOUBLE *f,
                              AATGS_DOUBLE *x_new,
                              AATGS_DOUBLE *x_old,
                              AATGS_DOUBLE *theta,
                              AATGS_DOUBLE *t, 
                              AATGS_DOUBLE beta,
                              aatgs_optimizer_anderson_restart_type restart_type,
                              AATGS_DOUBLE safeguard,
                              AATGS_DOUBLE *xrec,
                              AATGS_DOUBLE tol_orth, 
                              AATGS_DOUBLE tol_reorth)
{
   AATGS_DOUBLE *p = U + idx*n;
   AATGS_DOUBLE *v = Q + idx*n;
   AATGS_DOUBLE *q = NULL, *u = NULL, *qj = NULL, *uj = NULL;
   AATGS_DOUBLE sij;
   
   int i, j;
   AATGS_DOUBLE t1, normq, s = 0.0, normxrec = 0.0;

   /*------------------------
    * 1: Update Q and U
    *------------------------*/

   /* Compute ||v|| */
   *t = AatgsVecNorm2( v, n);

   // TODO: can save some time if do differently for different restart type and use OpenMP
   xrec[idx] = 0.0;
   for(i = 0 ; i < n ; i ++)
   {
      AATGS_MAX(xrec[idx], fabs(p[i]), xrec[idx]);
   }

   if( *t >= AATGS_EPS )
   {
      t1 = 1.0 / *t;
      AatgsVecScale( v, n, t1);
      AatgsVecScale( p, n, t1);
   }
   else
   {
      printf("Error: breakdown in Reverse Gram-Schmidt orthogonalization.\n");
      return -1;
   }

   /*------------------------
    * 2: reverse CGS2 step
    *------------------------*/
   theta[idx] = AatgsVecDdot(f, n, v);

   int idxi = idx - 1;

   for(i = 0 ; i < cwsize ; i ++)
   {
      idxi = idxi < 0 ? ncol-1 : idxi;

      q = Q + idxi*n;
      u = U + idxi*n;

      int idxj = idx;
      for(j = 0 ; j <= i ; j ++)
      {
         idxj = idxj < 0 ? ncol-1 : idxj;
         
         qj = Q + idxj*n;
         uj = U + idxj*n;

         t1 = AatgsVecDdot(q, n, qj);
         AatgsVecAxpy( -t1, uj, n, u);
         AatgsVecAxpy( -t1, qj, n, q);
         //xrec[idxi] += t1*fabs(xrec[idxj]);
         t1 = AatgsVecDdot(q, n, qj);
         AatgsVecAxpy( -t1, uj, n, u);
         AatgsVecAxpy( -t1, qj, n, q);
         //xrec[idxi] += t1*fabs(xrec[idxj]);

         idxj--;
      }

      normq = AatgsVecNorm2( q, n);
      if(normq > AATGS_EPS)
      //if(normq > 1e-03)
      {
         t1 = 1.0 / normq;
         AatgsVecScale( u, n, t1);
         AatgsVecScale( q, n, t1);
         //s += xrec[idxi] * t1;
         s += xrec[idxi];
         normxrec += xrec[idxi]*xrec[idxi];
      }
      else
      {
         printf("Warning: useless previous vectors.\n");
         AatgsVecFill( u, n, 0.0);
         AatgsVecFill( q, n, 0.0);
      }
      idxi--;
   }

   /*------------------------
    * 3: Compute theta
    *------------------------*/

   idxi = idx - 1;
   for(i = 0 ; i < cwsize ; i ++)
   {
      idxi = idxi < 0 ? ncol-1 : idxi;
      q = Q + idxi*n;
      
      theta[idxi] = AatgsVecDdot(f, n, q);

      idxi--;
   }

   /*------------------------
    * 4: Update theta x
    *------------------------*/

   int flag = 0;

   AATGS_MEMCPY(x_new, x_old, n, AATGS_DOUBLE);
   AatgsVecAxpy(beta, f, n, x_new);
   idxi = idx;
   for(int i = 0 ; i <= cwsize ; i ++)
   {
      idxi = idxi < 0 ? ncol-1 : idxi;
      q = Q + idxi*n;
      u = U + idxi*n;

      AatgsVecAxpy( -theta[idxi], u, n, x_new);
      AatgsVecAxpy( -beta*theta[idxi], q, n, x_new);

      /*
      if(cwsize >= 3 && i == cwsize)
      {
         if(fabs(theta[idxi])/fabs(theta[idx]) > 1)
         {
            flag = 1;
         }
      }
      */

      idxi--;
   }

   /*------------------------
    * 6: Update xrec
    *------------------------*/

   if(cwsize == 0)
   {
      xrec[idx] = xrec[idx] / *t;
   }
   else
   {
      xrec[idx] = (s + xrec[idx]) / *t;
      //xrec[idx] = s + xrec[idx] / *t;
   }

   if(restart_type == AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NORM_BOUND)
   {
      normxrec = sqrt(normxrec);
      if(normxrec > safeguard)
      {
         return 1;
      }
   }
   else
   {
      return 0;
   }

   return 0;
}

/**
 * @brief   Anderson acceleration with Reversed Gram-Schmidt orthogonalization.
 * @details Anderson acceleration with Reversed Gram-Schmidt orthogonalization.
 * @note    Helper function.
 * @param[in]     voptimizer  Pointer to the optimizer.
 * @param[in]     vproblem    Pointer to the problem.
 * @param[in]     x           Point at which the loss function and its gradient are computed.
 * @param[out]    x_final     Final point.
 * @return  0 if maxits reached, 1 if tol reached, -1 if error occured.
 */
int AatgsOptimizationAndersonRunAARTGS( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_anderson anderson = (paatgs_optimizer_anderson)optimizer->_optimizer_data;
   paatgs_problem problem = (paatgs_problem)vproblem;

   /* Initialize GD */
   AATGS_CALLOC(anderson->_loss_history, anderson->_maxits+1, AATGS_DOUBLE);
   if(anderson->_keep_x_history)
   {
      AATGS_CALLOC(anderson->_x_history, (size_t)(anderson->_maxits+1)*anderson->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(anderson->_x_history, 2*anderson->_n, AATGS_DOUBLE);
   }
   if(anderson->_keep_grad_history)
   {
      AATGS_CALLOC(anderson->_grad_history, (size_t)(anderson->_maxits+1)*anderson->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(anderson->_grad_history, anderson->_n, AATGS_DOUBLE);
   }
   AATGS_CALLOC(anderson->_grad_norm_history, anderson->_maxits+1, AATGS_DOUBLE);

   int wsize = anderson->_wsize;
   if(wsize <= 0)
   {
      wsize = anderson->_maxits;
   }
   AATGS_MIN(wsize, anderson->_maxits, wsize);

   if(wsize <= 0)
   {
      printf("Error: window size is invalid.\n");
      return -1;
   }

   AATGS_DOUBLE *Q = NULL, *p = NULL;
   AATGS_DOUBLE *U = NULL, *v = NULL;
   AATGS_MALLOC(Q, (size_t)(wsize+1)*anderson->_n, AATGS_DOUBLE);
   v = Q;
   AATGS_MALLOC(U, (size_t)(wsize+1)*anderson->_n, AATGS_DOUBLE);
   p = U;

   AATGS_DOUBLE *x1 = anderson->_x_history;
   AATGS_DOUBLE *x2 = anderson->_x_history + anderson->_n;
   AATGS_DOUBLE *x3 = NULL;
   AATGS_DOUBLE *xgrad = anderson->_grad_history;

   AATGS_DOUBLE *f = NULL, *f1 = NULL, *f2 = NULL, *f3 = NULL;
   AATGS_MALLOC(f, (size_t)2*anderson->_n, AATGS_DOUBLE);
   f1 = f;
   f2 = f + anderson->_n;

   AATGS_DOUBLE *theta = NULL;
   AATGS_MALLOC(theta, wsize+1, AATGS_DOUBLE);

   AATGS_MEMCPY(x1, x, anderson->_n, AATGS_DOUBLE);

   printf("Iteration  | Loss            | Grad norm     \n");

   int i, err = 0;
   int idx = 0, pidx = 0, vidx = 0, psize = 0, vsize = 0;
   AATGS_DOUBLE t, s;
   AATGS_DOUBLE *xrec = NULL;
   AATGS_MALLOC(xrec, wsize+1, AATGS_DOUBLE);
   int flag = 0;

   /* LOOP ANDERSON */
   
   int restart = 0;
   for(anderson->_nits = 0 ; anderson->_nits <= anderson->_maxits ; anderson->_nits++)
   {
      // compute the loss function and its gradient
      err = problem->_loss(vproblem, x1, anderson->_loss_history+anderson->_nits, f1);
      
      if(err != 0)
      {
         printf("%10d | Error: loss function failed\n", anderson->_nits);
         AATGS_MEMCPY(x_final, x1, anderson->_n, AATGS_DOUBLE);
         return -1;
      }
      anderson->_grad_norm_history[anderson->_nits] = AatgsVecNorm2( f1, anderson->_n);

      if(anderson->_grad_history)
      {
         AATGS_MEMCPY(xgrad, f1, anderson->_n, AATGS_DOUBLE);
      }
      AatgsVecScale( f1, anderson->_n, -anderson->_lr);

      if(psize == 0)
      {
         AATGS_MEMCPY(p, f1, anderson->_n, AATGS_DOUBLE);
         AatgsVecScale( p, anderson->_n, anderson->_beta);
#ifdef AATGS_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
            for(i = 0 ; i < anderson->_n ; i ++)
            {
               x2[i] = x1[i] + p[i];
            }
         }
         else
         {
#endif
            for(i = 0 ; i < anderson->_n ; i ++)
            {
               x2[i] = x1[i] + p[i];
            }
#ifdef AATGS_USING_OPENMP
         }
#endif
         pidx++;
         pidx = pidx % (wsize+1);
         p = U + (size_t)pidx*anderson->_n;
         psize ++;
      }
      else
      {
#ifdef AATGS_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
            for(i = 0 ; i < anderson->_n ; i ++)
            {
               v[i] = f1[i] - f2[i];
            }
         }
         else
         {
#endif
            for(i = 0 ; i < anderson->_n ; i ++)
            {
               v[i] = f1[i] - f2[i];
            }
#ifdef AATGS_USING_OPENMP
         }
#endif
         vidx++;
         vidx = vidx % (wsize+1);
         v = Q + (size_t)vidx*anderson->_n;
         
         flag = AatgsRGSAnderson( U, Q, anderson->_n, wsize+1, vsize, idx, f1, x2, x1, theta, &t, 
                                    anderson->_beta, anderson->_restart_type, anderson->_safeguard, xrec, AATGS_EPS, 0.7071);

         if(anderson->_restart > 0 && restart > anderson->_restart - 1)
         {
            restart = 0;
            printf("%10d | Restarting: iteration count met\n", anderson->_nits);
            vsize = 0;
            psize = 0;
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
            }
            else
            {
#endif
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
            pidx++;
            pidx = pidx % (wsize+1);
            p = U + (size_t)pidx*anderson->_n;
            psize++;
            AATGS_MIN(psize, wsize, psize);

            idx++;
            idx = idx % (wsize+1);
         }
         else if(flag == 0)
         {
            vsize ++;
            AATGS_MIN(vsize, wsize, vsize);
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
            }
            else
            {
#endif
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
            pidx++;
            pidx = pidx % (wsize+1);
            p = U + (size_t)pidx*anderson->_n;
            psize++;
            AATGS_MIN(psize, wsize, psize);

            idx++;
            idx = idx % (wsize+1);
         }
         else if(flag == 1)
         {
            restart = 0;
            printf("%10d | Restarting: restart condition\n", anderson->_nits);
            vsize = 0;
            psize = 0;
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
            }
            else
            {
#endif
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
            pidx++;
            pidx = pidx % (wsize+1);
            p = U + (size_t)pidx*anderson->_n;
            psize++;
            AATGS_MIN(psize, wsize, psize);

            idx++;
            idx = idx % (wsize+1);
         }
         else
         {
            printf("%10d | Restarting: linearly dependent basis\n", anderson->_nits);
            psize = 0;
            vsize = 0;
            AATGS_MEMCPY(p, f1, anderson->_n, AATGS_DOUBLE);
            AatgsVecScale( p, anderson->_n, anderson->_beta);
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  x2[i] = x1[i] + p[i];
               }
            }
            else
            {
#endif
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  x2[i] = x1[i] + p[i];
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
            pidx++;
            pidx = pidx % (wsize+1);
            p = U + (size_t)pidx*anderson->_n;
            psize ++;

            idx++;
            idx = idx % (wsize+1);
         }
      }

      // check the stopping criteria
      // print the information for this iteration
      // we have 3 columns: iteration | loss | grad_norm
      // each should be 15 characters long
      // with vertical bars

      printf("%10d | %15.8e | %15.8e\n", anderson->_nits, anderson->_loss_history[anderson->_nits], anderson->_grad_norm_history[anderson->_nits]);

      if(anderson->_grad_norm_history[anderson->_nits] < anderson->_tol)
      {
         printf("Iteration stopped at %d with norm %e\n", anderson->_nits, anderson->_grad_norm_history[anderson->_nits]);
         AATGS_MEMCPY(x_final, x2, anderson->_n, AATGS_DOUBLE);
         return 1; // tol reached
      }

      f3 = f1;
      f1 = f2;
      f2 = f3;

      if(anderson->_keep_x_history)
      {
         x1 += anderson->_n;
         x2 += anderson->_n;
      }
      else
      {
         x3 = x1;
         x1 = x2;
         x2 = x3;
      }

      if(anderson->_keep_grad_history)
      {
         xgrad += anderson->_n;
      }

      restart++;
   }// end of main loop

   printf("Maximum number of iterations reached with norm %e\n", anderson->_grad_norm_history[anderson->_nits-1]);
   AATGS_MEMCPY(x_final, x2, anderson->_n, AATGS_DOUBLE);

   AATGS_FREE(Q);
   AATGS_FREE(U);
   AATGS_FREE(f);
   AATGS_FREE(theta);
   AATGS_FREE(xrec);

   return 0;
}

/**
 * @brief   Modified Gram-Schmidt orthogonalization for Anderson.
 * @details Modified Gram-Schmidt orthogonalization for Anderson.
 * @note    Helper function.
 * @param[in,out] U              Pointer to the U matrix.
 * @param[in,out] Q              Pointer to the Q matrix.
 * @param[in]     n              Dimension of the vector.
 * @param[in]     ncol           Number of columns of the two matrices.
 * @param[in]     cwsize         Current window size used.
 * @param[in]     idx            Index of the current vector.
 * @param[out]    f              Pointer to the f at current iteration.
 * @param[in,out] x_new          Pointer to the new x at current iteration.
 * @param[in]     x_old          Pointer to the old x at current iteration.
 * @param[in,out] theta          Pointer to the theta at current iteration.
 * @param[out]    t              Vector norm of the crruent vector after the orthogonalization.
 * @param[in]     beta           Mixing parameter.
 * @param[in]     restart_type   Restart type.
 * @param[in]     safeguard      Restart parameter.
 * @param[in,out] xrec           Pointer to the xrec at current iteration.
 * @param[in,out] r              Pointer to the r at current iteration.
 * @param[in,out] s              Pointer to the s at current iteration.
 * @param[in]     tol_orth       Tolerance for the orthogonality.
 * @param[in]     tol_reorth     Tolerance for the reorthogonalization.
 * @return  0 if success.
 */
int AatgsModifiedGSAnderson( AATGS_DOUBLE *U, 
                              AATGS_DOUBLE *Q, 
                              int n, 
                              int ncol, 
                              int cwsize, 
                              int idx, 
                              AATGS_DOUBLE *f,
                              AATGS_DOUBLE *x_new,
                              AATGS_DOUBLE *x_old,
                              AATGS_DOUBLE *theta,
                              AATGS_DOUBLE *t, 
                              AATGS_DOUBLE beta,
                              aatgs_optimizer_anderson_restart_type restart_type,
                              AATGS_DOUBLE safeguard,
                              AATGS_DOUBLE *xrec,
                              AATGS_DOUBLE *s,
                              AATGS_DOUBLE tol_orth, 
                              AATGS_DOUBLE tol_reorth)
{
   AATGS_DOUBLE *p = U + idx*n;
   AATGS_DOUBLE *v = Q + idx*n;
   AATGS_DOUBLE *q = NULL, *u = NULL;
   AATGS_DOUBLE sij;
   
   int i;
   AATGS_DOUBLE t0, t1, normv, normxrec;

   /* compute initial ||w|| if we need to reorth */
   if(tol_reorth > 0.0)
   {
      normv = AatgsVecNorm2(v, n);
   }
   else
   {
      normv = 0.0;
   }

   *s = 0.0;
   normxrec = 0.0;
   // TODO: can save some time if do differently for different restart type and use OpenMP
   t0 = 0.0;
   for(i = 0 ; i < n ; i ++)
   {
      AATGS_MAX(t0, fabs(p[i]), t0);
   }
   
   int idxi = idx - 1;
   for(i = 0 ; i < cwsize ; i ++)
   {
      idxi = idxi < 0 ? ncol-1 : idxi;
      q = Q + idxi*n;
      u = U + idxi*n;

      t1 = AatgsVecDdot(v, n, q);
      // TODO: can save some time if do differently for different restart type
      *s += xrec[idxi] * t1;
      normxrec += xrec[idxi] * xrec[idxi];
      theta[idxi] = AatgsVecDdot(f, n, q);

      AatgsVecAxpy( -t1, u, n, p);
      AatgsVecAxpy( -t1, q, n, v);
      idxi--;
   }

   /* Compute ||v|| */
   *t = AatgsVecNorm2( v, n);

   /*------------------------
    * 2: Re-orth step
    *------------------------*/

   /* t < tol_orth is considered be lucky breakdown */
   while( *t < normv * tol_reorth && *t >= tol_orth)
   {
      normv = *t;
      /* Re-orth */
      idxi = idx - 1;
      for(i = 0 ; i < cwsize ; i ++)
      {
         idxi = idxi < 0 ? ncol-1 : idxi;
         q = Q + idxi*n;
         u = U + idxi*n;

         t1 = AatgsVecDdot(v, n, q);
         // TODO: can save some time if do differently for different restart type
         *s += xrec[idxi] * t1;
         normxrec += xrec[idxi] * xrec[idxi];

         AatgsVecAxpy( -t1, u, n, p);
         AatgsVecAxpy( -t1, q, n, v);
         idxi--;
      }
      /* Compute ||v|| */
      *t = AatgsVecNorm2( v, n);
   }

   /*------------------------
    * 3: Update Q and U
    *------------------------*/
   if( *t >= AATGS_EPS )
   {
      t1 = 1.0 / *t;
      AatgsVecScale( v, n, t1);
      AatgsVecScale( p, n, t1);
   }
   else
   {
      //printf("Error: breakdown in Modified Gram-Schmidt orthogonalization.\n");
      return -1;
   }

   /*------------------------
    * 4: Compute theta idx
    *------------------------*/
   theta[idx] = AatgsVecDdot(f, n, v);

   /*------------------------
    * 5: Update theta x
    *------------------------*/
   AATGS_MEMCPY(x_new, x_old, n, AATGS_DOUBLE);
   AatgsVecAxpy(beta, f, n, x_new);
   idxi = idx;
   for(int i = 0 ; i <= cwsize ; i ++)
   {
      idxi = idxi < 0 ? ncol-1 : idxi;
      q = Q + idxi*n;
      u = U + idxi*n;

      AatgsVecAxpy( -theta[idxi], u, n, x_new);
      AatgsVecAxpy( -beta*theta[idxi], q, n, x_new);
      idxi--;
   }

   /*------------------------
    * 6: Update xrec
    *------------------------*/

   if(cwsize == 0)
   {
      *s = t0 / *t;
   }
   else
   {
      *s = (*s + t0) / *t;
   }
   xrec[idx] = *s;

   if(restart_type == AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NORM_BOUND)
   {
      normxrec = sqrt(normxrec + *s * *s);
      if(normxrec > safeguard)
      {
         return 1;
      }
   }

   return 0;
}

/**
 * @brief   Anderson acceleration with Gram-Schmidt orthogonalization.
 * @details Anderson acceleration with Gram-Schmidt orthogonalization.
 * @note    Helper function.
 * @param[in]     voptimizer  Pointer to the optimizer.
 * @param[in]     vproblem    Pointer to the problem.
 * @param[in]     x           Point at which the loss function and its gradient are computed.
 * @param[out]    x_final     Final point.
 * @return  0 if maxits reached, 1 if tol reached, -1 if error occured.
 */
int AatgsOptimizationAndersonRunAATGS( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_anderson anderson = (paatgs_optimizer_anderson)optimizer->_optimizer_data;
   paatgs_problem problem = (paatgs_problem)vproblem;

   /* Initialize GD */
   AATGS_CALLOC(anderson->_loss_history, anderson->_maxits+1, AATGS_DOUBLE);
   if(anderson->_keep_x_history)
   {
      AATGS_CALLOC(anderson->_x_history, (size_t)(anderson->_maxits+1)*anderson->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(anderson->_x_history, 2*anderson->_n, AATGS_DOUBLE);
   }
   if(anderson->_keep_grad_history)
   {
      AATGS_CALLOC(anderson->_grad_history, (size_t)(anderson->_maxits+1)*anderson->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(anderson->_grad_history, anderson->_n, AATGS_DOUBLE);
   }
   AATGS_CALLOC(anderson->_grad_norm_history, anderson->_maxits+1, AATGS_DOUBLE);

   int wsize = anderson->_wsize;
   if(wsize <= 0)
   {
      wsize = anderson->_maxits;
   }
   AATGS_MIN(wsize, anderson->_maxits, wsize);

   if(wsize <= 0)
   {
      printf("Error: window size is invalid.\n");
      return -1;
   }

   AATGS_DOUBLE *Q = NULL, *p = NULL;
   AATGS_DOUBLE *U = NULL, *v = NULL;
   AATGS_MALLOC(Q, (size_t)(wsize+1)*anderson->_n, AATGS_DOUBLE);
   v = Q;
   AATGS_MALLOC(U, (size_t)(wsize+1)*anderson->_n, AATGS_DOUBLE);
   p = U;

   AATGS_DOUBLE *x1 = anderson->_x_history;
   AATGS_DOUBLE *x2 = anderson->_x_history + anderson->_n;
   AATGS_DOUBLE *x3 = NULL;
   AATGS_DOUBLE *xgrad = anderson->_grad_history;

   AATGS_DOUBLE *f = NULL, *f1 = NULL, *f2 = NULL, *f3 = NULL;
   AATGS_MALLOC(f, (size_t)2*anderson->_n, AATGS_DOUBLE);
   f1 = f;
   f2 = f + anderson->_n;

   AATGS_DOUBLE *theta = NULL;
   AATGS_MALLOC(theta, wsize+1, AATGS_DOUBLE);

   AATGS_MEMCPY(x1, x, anderson->_n, AATGS_DOUBLE);

   printf("Iteration  | Loss            | Grad norm     \n");

   int i, err = 0;
   int idx = 0, pidx = 0, vidx = 0, psize = 0, vsize = 0;
   AATGS_DOUBLE t, s;
   AATGS_DOUBLE *xrec = NULL;
   AATGS_MALLOC(xrec, wsize+1, AATGS_DOUBLE);
   int flag = 0;

   /* LOOP ANDERSON */
   int restart = 0;
   for(anderson->_nits = 0 ; anderson->_nits <= anderson->_maxits ; anderson->_nits++)
   {
      // compute the loss function and its gradient
      err = problem->_loss(vproblem, x1, anderson->_loss_history+anderson->_nits, f1);
      
      if(err != 0)
      {
         printf("%10d | Error: loss function failed\n", anderson->_nits);
         AATGS_MEMCPY(x_final, x1, anderson->_n, AATGS_DOUBLE);
         return -1;
      }
      anderson->_grad_norm_history[anderson->_nits] = AatgsVecNorm2( f1, anderson->_n);

      if(anderson->_grad_history)
      {
         AATGS_MEMCPY(xgrad, f1, anderson->_n, AATGS_DOUBLE);
      }
      AatgsVecScale( f1, anderson->_n, -anderson->_lr);

      if(psize == 0)
      {
         AATGS_MEMCPY(p, f1, anderson->_n, AATGS_DOUBLE);
         AatgsVecScale( p, anderson->_n, anderson->_beta);
#ifdef AATGS_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
            for(i = 0 ; i < anderson->_n ; i ++)
            {
               x2[i] = x1[i] + p[i];
            }
         }
         else
         {
#endif
            for(i = 0 ; i < anderson->_n ; i ++)
            {
               x2[i] = x1[i] + p[i];
            }
#ifdef AATGS_USING_OPENMP
         }
#endif
         pidx++;
         pidx = pidx % (wsize+1);
         p = U + (size_t)pidx*anderson->_n;
         psize ++;
      }
      else
      {
#ifdef AATGS_USING_OPENMP
         if(!omp_in_parallel())
         {
            #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
            for(i = 0 ; i < anderson->_n ; i ++)
            {
               v[i] = f1[i] - f2[i];
            }
         }
         else
         {
#endif
            for(i = 0 ; i < anderson->_n ; i ++)
            {
               v[i] = f1[i] - f2[i];
            }
#ifdef AATGS_USING_OPENMP
         }
#endif
         vidx++;
         vidx = vidx % (wsize+1);
         v = Q + (size_t)vidx*anderson->_n;
         
         flag = AatgsModifiedGSAnderson( U, Q, anderson->_n, wsize+1, vsize, idx, f1, x2, x1, theta, &t, 
                                          anderson->_beta, anderson->_restart_type, anderson->_safeguard, xrec, &s,
                                          AATGS_EPS, 0.7071);
         if(anderson->_restart > 0 && restart > anderson->_restart - 1)
         {
            restart = 0;
            printf("%10d | Restarting: iteration count met\n", anderson->_nits);
            vsize = 0;
            psize = 0;
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
            }
            else
            {
#endif
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
            pidx++;
            pidx = pidx % (wsize+1);
            p = U + (size_t)pidx*anderson->_n;
            psize++;
            AATGS_MIN(psize, wsize, psize);

            idx++;
            idx = idx % (wsize+1);
         }
         else if(flag == 0)
         {
            vsize ++;
            AATGS_MIN(vsize, wsize, vsize);
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
            }
            else
            {
#endif
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
            pidx++;
            pidx = pidx % (wsize+1);
            p = U + (size_t)pidx*anderson->_n;
            psize++;
            AATGS_MIN(psize, wsize, psize);

            idx++;
            idx = idx % (wsize+1);
         }
         else if(flag == 1)
         {
            restart = 0;
            printf("%10d | Restarting: restart condition\n", anderson->_nits);
            vsize = 0;
            psize = 0;
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
            }
            else
            {
#endif
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  p[i] = x2[i] - x1[i];
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
            pidx++;
            pidx = pidx % (wsize+1);
            p = U + (size_t)pidx*anderson->_n;
            psize++;
            AATGS_MIN(psize, wsize, psize);

            idx++;
            idx = idx % (wsize+1);
         }
         else
         {
            printf("%10d | Restarting: linearly dependent basis\n", anderson->_nits);
            psize = 0;
            vsize = 0;
            AATGS_MEMCPY(p, f1, anderson->_n, AATGS_DOUBLE);
            AatgsVecScale( p, anderson->_n, anderson->_beta);
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  x2[i] = x1[i] + p[i];
               }
            }
            else
            {
#endif
               for(i = 0 ; i < anderson->_n ; i ++)
               {
                  x2[i] = x1[i] + p[i];
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
            pidx++;
            pidx = pidx % (wsize+1);
            p = U + (size_t)pidx*anderson->_n;
            psize ++;

            idx++;
            idx = idx % (wsize+1);
         }
      }

      // check the stopping criteria
      // print the information for this iteration
      // we have 3 columns: iteration | loss | grad_norm
      // each should be 15 characters long
      // with vertical bars

      printf("%10d | %15.8e | %15.8e\n", anderson->_nits, anderson->_loss_history[anderson->_nits], anderson->_grad_norm_history[anderson->_nits]);

      if(anderson->_grad_norm_history[anderson->_nits] < anderson->_tol)
      {
         printf("Iteration stopped at %d with norm %e\n", anderson->_nits, anderson->_grad_norm_history[anderson->_nits]);
         AATGS_MEMCPY(x_final, x2, anderson->_n, AATGS_DOUBLE);
         return 1; // tol reached
      }

      f3 = f1;
      f1 = f2;
      f2 = f3;

      if(anderson->_keep_x_history)
      {
         x1 += anderson->_n;
         x2 += anderson->_n;
      }
      else
      {
         x3 = x1;
         x1 = x2;
         x2 = x3;
      }

      if(anderson->_keep_grad_history)
      {
         xgrad += anderson->_n;
      }
      restart++;
   }// end of main loop

   printf("Maximum number of iterations reached with norm %e\n", anderson->_grad_norm_history[anderson->_nits-1]);
   AATGS_MEMCPY(x_final, x2, anderson->_n, AATGS_DOUBLE);

   AATGS_FREE(Q);
   AATGS_FREE(U);
   AATGS_FREE(f);
   AATGS_FREE(theta);
   AATGS_FREE(xrec);

   return 0;
}

int AatgsOptimizationAndersonRun( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final)
{
   /* set problem */
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_anderson anderson = (paatgs_optimizer_anderson)optimizer->_optimizer_data;
   paatgs_problem problem = (paatgs_problem)vproblem;
   anderson->_problem = problem;
   anderson->_n = problem->_n;

   if(anderson->_nits > 0)
   {
      printf("Error: cannot rerun optimization.\n");
      return -1;
   }

   switch(anderson->_type)
   {
      case AATGS_OPTIMIZER_ANDERSON_TYPE_AATGS:
      {
         printf("============================================\n");
         printf("Running Standard AATGS\n");
         printf("Window size (history vector used in orthogonalization): %d\n", anderson->_wsize);
         printf("Learning rate (mu): %e\n", anderson->_lr);
         printf("Mixing parameter (beta): %e\n", anderson->_beta);
         switch(anderson->_restart_type)
         {
            case AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NONE:
            {
               printf("Restart type: none\n");
               break;
            }
            case AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NORM_BOUND:
            {
               printf("Restart type: inf norm bound\n");
               break;
            }
            case AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_COOKTAIL:
            {
               printf("Restart type: cocktail\n");
               break;
            }
            default:
            {
               printf("Restart type: none\n");
               break;
            }
         }
         printf("============================================\n");
         return AatgsOptimizationAndersonRunAATGS(voptimizer, vproblem, x, x_final);
         break;
      }
      case AATGS_OPTIMIZER_ANDERSON_TYPE_AARTGS:
      {
         printf("============================================\n");
         printf("Running reverse-orth AARTGS\n");
         printf("Window size (history vector used in orthogonalization): %d\n", anderson->_wsize);
         printf("Learning rate (mu): %e\n", anderson->_lr);
         printf("Mixing parameter (beta): %e\n", anderson->_beta);
         switch(anderson->_restart_type)
         {
            case AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NONE:
            {
               printf("Restart type: none\n");
               break;
            }
            case AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NORM_BOUND:
            {
               printf("Restart type: inf norm bound\n");
               break;
            }
            case AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_COOKTAIL:
            {
               printf("Restart type: cocktail\n");
               break;
            }
            default:
            {
               printf("Restart type: none\n");
               break;
            }
         }
         printf("============================================\n");
         return AatgsOptimizationAndersonRunAARTGS(voptimizer, vproblem, x, x_final);
         break;
      }
      default:
      {
         printf("Error: unknown Anderson type, running GS Anderson.\n");
         return AatgsOptimizationAndersonRunAATGS(voptimizer, vproblem, x, x_final);
      }
   }


   return 0; // maxits reached
}

int AatgsOptimizationAndersonGetInfo( void *voptimizer, int *nitsp, AATGS_DOUBLE **xp, AATGS_DOUBLE **lossp, AATGS_DOUBLE **gradp, AATGS_DOUBLE **grad_normp)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_anderson anderson = (paatgs_optimizer_anderson)optimizer->_optimizer_data;

   if(nitsp)
   {
      *nitsp = anderson->_nits;
   }

   if(xp)
   {
      *xp = anderson->_x_history;
   }

   if(lossp)
   {
      *lossp = anderson->_loss_history;
   }

   if(gradp)
   {
      *gradp = anderson->_grad_history;
   }

   if(grad_normp)
   {
      *grad_normp = anderson->_grad_norm_history;
   }

   return 0;
}
