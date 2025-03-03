#include "float.h"
#include "nltgcr.h"
#include "../ops/vecops.h"
#include "../ops/matops.h"

void* AatgsOptimizationNltgcrCreate()
{
   paatgs_optimizer optimizer = NULL;
   AATGS_MALLOC(optimizer, 1, aatgs_optimizer);
   paatgs_optimizer_nltgcr nltgcr = NULL;
   AATGS_MALLOC(nltgcr, 1, aatgs_optimizer_nltgcr);

   nltgcr->_maxits = 1000;
   nltgcr->_tol = 1e-6;
   nltgcr->_type = AATGS_OPTIMIZER_NLTGCR_TYPE_NONLINEAR;
   nltgcr->_restart_type = AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NONE;

   nltgcr->_problem = NULL;

   nltgcr->_n = 0;

   nltgcr->_lr = 1.0;
   nltgcr->_agc = 0.001;
   nltgcr->_wsize = -1; // negative means use full size
   nltgcr->_restart = -1; // no mandatory restart
   nltgcr->_safeguard = 1e03;
   nltgcr->_complex_diff = 0;

   nltgcr->_nits = 0;
   nltgcr->_loss_history = NULL;
   nltgcr->_keep_x_history = 0;
   nltgcr->_x_history = NULL;
   nltgcr->_keep_grad_history = 0;
   nltgcr->_grad_history = NULL;
   nltgcr->_grad_norm_history = NULL;

   optimizer->_optimizer_data = nltgcr;
   optimizer->_run_optimizer = &AatgsOptimizationNltgcrRun;
   optimizer->_optimizer_get_info = &AatgsOptimizationNltgcrGetInfo;
   optimizer->_free_optimizer = &AatgsOptimizationNltgcrFree;

   return (void*)optimizer;
}

void AatgsOptimizationNltgcrFree( void **voptimizerp )
{
   if(*voptimizerp)
   {
      paatgs_optimizer optimizer = (paatgs_optimizer)*voptimizerp;
      paatgs_optimizer_nltgcr nltgcr = (paatgs_optimizer_nltgcr)optimizer->_optimizer_data;
      if(nltgcr)
      {
         AATGS_FREE(nltgcr->_loss_history);
         AATGS_FREE(nltgcr->_x_history);
         AATGS_FREE(nltgcr->_grad_history);
         AATGS_FREE(nltgcr->_grad_norm_history);
      }

      AATGS_FREE(nltgcr);
   }
   AATGS_FREE(*voptimizerp);
}

int AatgsOptimizationNltgcrSetOptions( void *voptimizer, 
                                    int maxits,
                                    AATGS_DOUBLE tol)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_nltgcr nltgcr = (paatgs_optimizer_nltgcr)optimizer->_optimizer_data;

   if(nltgcr->_nits > 0)
   {
      printf("Error: cannot set options after the first optimization step.\n");
      return -1;
   }

   nltgcr->_maxits = maxits;
   nltgcr->_tol = tol;

   return 0;
}

int AatgsOptimizationNltgcrSetParameters( void *voptimizer,
                                       AATGS_DOUBLE lr,
                                       int max_ls,
                                       int wsize,
                                       AATGS_DOUBLE agc,
                                       int complex_diff,
                                       int restart,
                                       AATGS_DOUBLE safeguard,
                                       aatgs_optimizer_nltgcr_type type,
                                       aatgs_optimizer_nltgcr_restart_type restart_type)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_nltgcr nltgcr = (paatgs_optimizer_nltgcr)optimizer->_optimizer_data;

   if(nltgcr->_nits > 0)
   {
      printf("Error: cannot set parameters after the first optimization step.\n");
      return -1;
   }

   nltgcr->_lr = lr;
   nltgcr->_max_ls = max_ls;
   nltgcr->_wsize = wsize;
   nltgcr->_agc = agc;
   nltgcr->_complex_diff = complex_diff;
   nltgcr->_safeguard = safeguard;
   nltgcr->_restart = restart;
   nltgcr->_type = type;
   nltgcr->_restart_type = restart_type;

   return 0;
}

int AatgsOptimizationNltgcrSetHistory( void *voptimizer,
                                      int keep_x_history,
                                      int keep_grad_history)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_nltgcr nltgcr = (paatgs_optimizer_nltgcr)optimizer->_optimizer_data;

   if(nltgcr->_nits > 0)
   {
      printf("Error: cannot set history after the first optimization step.\n");
      return -1;
   }

   nltgcr->_keep_x_history = keep_x_history;
   nltgcr->_keep_grad_history = keep_grad_history;

   return 0;
}

/**
 * @brief   Modified Gram-Schmidt orthogonalization for Anderson.
 * @details Modified Gram-Schmidt orthogonalization for Anderson.
 * @note    Helper function.
 * @param[in,out] V              Pointer to the V matrix.
 * @param[in,out] P              Pointer to the P matrix.
 * @param[in]     n              Dimension of the vector.
 * @param[in]     ncol           Number of columns of the two matrices.
 * @param[in]     vsize          Num of vectors in V orth aganist (exclude v and p)
 * @param[in]     vidx           Index of the current vector (the v and p)
 * @param[out]    t              Vector norm of the crruent vector after the orthogonalization.
 * @param[in]     safeguard      Tol for restart.
 * @param[in]     restart_type   Restart type.
 * @param[in,out] xrec           Pointer to the xrec at current iteration.
 * @param[in,out] r              Pointer to the r at current iteration.
 * @param[in]     tol_orth       Tolerance for the orthogonality.
 * @param[in]     tol_reorth     Tolerance for the reorthogonalization.
 * @return  0 if success.
 */
int AatgsModifiedGSNltgcr( AATGS_DOUBLE *V, 
                              AATGS_DOUBLE *P, 
                              int n, 
                              int ncol, 
                              int vsize, 
                              int vidx,
                              AATGS_DOUBLE *t, 
                              AATGS_DOUBLE safeguard,
                              aatgs_optimizer_nltgcr_restart_type restart_type,
                              AATGS_DOUBLE *xrec,
                              AATGS_DOUBLE tol_orth, 
                              AATGS_DOUBLE tol_reorth)
{
   AATGS_DOUBLE *p = P + vidx*n;
   AATGS_DOUBLE *v = V + vidx*n;
   AATGS_DOUBLE *pi = NULL, *vi = NULL;
   
   int i;
   AATGS_DOUBLE t1, normv;

   /* compute initial ||w|| if we need to reorth */
   if(tol_reorth > 0.0)
   {
      normv = AatgsVecNorm2(v, n);
   }
   else
   {
      normv = 0.0;
   }

   
   //printf("xvec pre %f | t pre %f |",xrec[vidx], *t);

   int idxi = vidx - 1;
   for(i = 0 ; i < vsize ; i ++)
   {
      idxi = idxi < 0 ? ncol-1 : idxi;
      pi = P + idxi*n;
      vi = V + idxi*n;

      t1 = AatgsVecDdot(v, n, vi);
      xrec[vidx] += xrec[idxi] * fabs(t1);

      AatgsVecAxpy( -t1, pi, n, p);
      AatgsVecAxpy( -t1, vi, n, v);
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
      idxi = vidx - 1;
      for(i = 0 ; i < vsize ; i ++)
      {
         idxi = idxi < 0 ? ncol-1 : idxi;
         pi = P + idxi*n;
         vi = V + idxi*n;

         t1 = AatgsVecDdot(v, n, vi);
         xrec[vidx] += xrec[idxi] * fabs(t1);

         AatgsVecAxpy( -t1, pi, n, p);
         AatgsVecAxpy( -t1, vi, n, v);
         idxi--;
      }
      /* Compute ||v|| */
      *t = AatgsVecNorm2( v, n);
   }

   /*------------------------
    * 3: Update Q and U
    *------------------------*/
   
   if( *t >= AATGS_EPS)
   {
      t1 = 1.0 / *t;
      AatgsVecScale( v, n, t1);
      AatgsVecScale( p, n, t1);
      xrec[vidx] *= t1;
   }
   else
   {
      //printf("Error: breakdown in Modified Gram-Schmidt orthogonalization.\n");
      return -1;
   }
   
   /*------------------------
    * 4: Update xrec
    *------------------------*/

   if(restart_type == AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NORM_BOUND)
   {
      //printf("Safe check: %f vs %f\n",xrec[vidx],safeguard);
      if( xrec[vidx] > safeguard)
      {
         return 1;
      }
   }

   return 0;
}

/**
 * @brief   Compute multiplication v = f'(x)*w.
 * @details Compute multiplication v = f'(x)*w.
 * @note    Helper function. Modified from dirder by C. T. Kelley, April 1, 2003
 * @param[in]     vproblem    Pointer to the problem struct
 * @param[in]     n           Length of the vector
 * @param[in]     x           Pointer to x
 * @param[in]     alpha       the actual w is alpha*w
 * @param[in]     w           the actual w is alpha*w
 * @param[in]     nrmw        2-norm of the fake w, set to -1.0 if don't know
 * @param[in]     beta        f(x) = beta*f0
 * @param[in]     f0          f(x) = beta*f0
 * @param[in]     dwork       Working buffer of size n
 * @param[out]    v           Pointer to the output.
 * @return  0 if success.
 */
int AatgsOptimizationNltgcrJVP( void *vproblem, int n, AATGS_DOUBLE *x, AATGS_DOUBLE alpha, AATGS_DOUBLE *w, AATGS_DOUBLE nrmw, AATGS_DOUBLE beta, AATGS_DOUBLE *f0, AATGS_DOUBLE *dwork, AATGS_DOUBLE *v)
{
   paatgs_problem problem = (paatgs_problem)vproblem;
#ifdef AATGS_USING_FLOAT32
   AATGS_DOUBLE epsnew = 1e-06;
#else
   AATGS_DOUBLE epsnew = 1e-07;
#endif

   // compute norm when needed
   nrmw = nrmw < 0.0 ? fabs(alpha) * AatgsVecNorm2(w, n) : fabs(alpha) * nrmw;
   AATGS_DOUBLE xs = alpha * AatgsVecDdot(x, n, w) / nrmw;

   if(xs != 0.0)
   {
      AATGS_DOUBLE signxs;
      AATGS_SIGN(xs, signxs);
      AATGS_MAX(fabs(xs), 1.0, xs);
      epsnew *= xs * signxs;
   }
   epsnew /= nrmw;
   epsnew *= alpha;

   int i;

#ifdef AATGS_USING_OPENMP
   if(!omp_in_parallel())
   {
      #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
      for(i = 0 ; i < n ; i ++)
      {
         dwork[i] = x[i] + epsnew * w[i];
      }
   }
   else
   {
#endif
      for(i = 0 ; i < n ; i ++)
      {
         dwork[i] = x[i] + epsnew * w[i];
      }
#ifdef AATGS_USING_OPENMP
   }
#endif

   int err = problem->_loss(vproblem, dwork, NULL, v);
   if(err != 0)
   {
      return -1;
   }

   AatgsVecAxpy(-beta, f0, n, v);
   AatgsVecScale(v, n, 1.0/epsnew);

   return 0;
}

/**
 * @brief   Nonlinear version NLTGCR.
 * @details Nonlinear version NLTGCR.
 * @note    Helper function.
 * @param[in]     voptimizer  Pointer to the optimizer.
 * @param[in]     vproblem    Pointer to the problem.
 * @param[in]     x           Point at which the loss function and its gradient are computed.
 * @param[out]    x_final     Final point.
 * @return  0 if maxits reached, 1 if tol reached, -1 if error occured.
 */
int AatgsOptimizationNltgcrRunNonlinear( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_nltgcr nltgcr = (paatgs_optimizer_nltgcr)optimizer->_optimizer_data;
   paatgs_problem problem = (paatgs_problem)vproblem;

   /* Initialize GD */
   AATGS_CALLOC(nltgcr->_loss_history, nltgcr->_maxits+1, AATGS_DOUBLE);
   if(nltgcr->_keep_x_history)
   {
      AATGS_CALLOC(nltgcr->_x_history, (size_t)(nltgcr->_maxits+1)*nltgcr->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(nltgcr->_x_history, 2*nltgcr->_n, AATGS_DOUBLE);
   }
   if(nltgcr->_keep_grad_history)
   {
      AATGS_CALLOC(nltgcr->_grad_history, (size_t)(nltgcr->_maxits+1)*nltgcr->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(nltgcr->_grad_history, nltgcr->_n, AATGS_DOUBLE);
   }
   AATGS_CALLOC(nltgcr->_grad_norm_history, nltgcr->_maxits+1, AATGS_DOUBLE);

   int wsize = nltgcr->_wsize;
   if(wsize <= 0)
   {
      wsize = nltgcr->_maxits;
   }
   AATGS_MIN(wsize, nltgcr->_maxits, wsize);

   if(wsize <= 0)
   {
      printf("Error: window size is invalid.\n");
      return -1;
   }

   AATGS_DOUBLE *P = NULL, *p = NULL;
   AATGS_DOUBLE *V = NULL, *v = NULL;
   AATGS_MALLOC(P, (size_t)(wsize+1)*nltgcr->_n, AATGS_DOUBLE);
   p = P;
   AATGS_MALLOC(V, (size_t)(wsize+1)*nltgcr->_n, AATGS_DOUBLE);
   v = V;

   AATGS_DOUBLE *x1 = nltgcr->_x_history;
   AATGS_DOUBLE *x2 = nltgcr->_x_history + nltgcr->_n;
   AATGS_DOUBLE *x3 = NULL;
   AATGS_DOUBLE *xgrad = nltgcr->_grad_history;

   AATGS_DOUBLE *r = NULL;
   AATGS_MALLOC(r, (size_t)2*nltgcr->_n, AATGS_DOUBLE);
   AATGS_DOUBLE *gg = r + nltgcr->_n;
   
   AATGS_DOUBLE *alph = NULL;
   AATGS_MALLOC(alph, wsize+1, AATGS_DOUBLE);

   AATGS_MEMCPY(x1, x, nltgcr->_n, AATGS_DOUBLE);

   printf("Iteration  | Loss            | Grad norm       | NFE\n");

   AATGS_DOUBLE lam0 = nltgcr->_lr;
   AATGS_DOUBLE tc = 0.0;

   int i, err = 0;
   int vidx = 0, vsize = 0;
   AATGS_DOUBLE t, s;
   AATGS_DOUBLE *xrec = NULL;
   AATGS_MALLOC(xrec, wsize+1, AATGS_DOUBLE);
   int flag = 0;
   int nfe = 0;

   nltgcr->_nits = 0;
   
   err = problem->_loss(vproblem, x1, nltgcr->_loss_history+nltgcr->_nits, r);nfe++;
   if(err != 0)
   {
      printf("%10d | Error: loss function failed\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return -1;
   }
   nltgcr->_grad_norm_history[nltgcr->_nits] = AatgsVecNorm2( r, nltgcr->_n);

   if(nltgcr->_grad_history)
   {
      AATGS_MEMCPY(xgrad, r, nltgcr->_n, AATGS_DOUBLE);
   }
   AatgsVecScale( r, nltgcr->_n, -1.0);

   if(nltgcr->_grad_norm_history[nltgcr->_nits] < nltgcr->_tol)
   {
      printf("%10d | Converge: gradient norm reached\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return 1;
   }

   err = AatgsOptimizationNltgcrJVP( vproblem, nltgcr->_n, x1, 1.0, r, 
                                 nltgcr->_grad_norm_history[nltgcr->_nits], -1.0, r, x2, v);nfe++;

   if(err != 0)
   {
      printf("%10d | Error: loss function failed\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return -1;
   }

   t = AatgsVecNorm2(v, nltgcr->_n);
   if(t < AATGS_EPS)
   {
      printf("%10d | Warning: JVP is too small\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return -1;
   }

   t = 1.0/t;
   AATGS_MEMCPY( p, r, nltgcr->_n, AATGS_DOUBLE);
   AatgsVecScale(p, nltgcr->_n, t);
   AatgsVecScale(v, nltgcr->_n, t);

   // TODO: can save some time if do differently for different restart type and use OpenMP
   xrec[vidx] = 0.0;
   for(i = 0 ; i < nltgcr->_n ; i ++)
   {
      AATGS_MAX(xrec[vidx], fabs(r[i]), xrec[vidx]);
   }
   
   vsize ++;

   printf("%10d | %15.8e | %15.8e | %5d\n", nltgcr->_nits, nltgcr->_loss_history[nltgcr->_nits], nltgcr->_grad_norm_history[nltgcr->_nits], nfe);

   /* LOOP NLTGCR */
   int restart = 0;
   nltgcr->_nits++;
   for(; nltgcr->_nits <= nltgcr->_maxits ; nltgcr->_nits++)
   {
      int mvsize = vidx+1-vsize;
      if(mvsize >= 0)
      {
         AatgsDenseMatGemv( V+(size_t)mvsize*nltgcr->_n, 'T', nltgcr->_n, vsize, 1.0, r, 0.0, alph);
         AatgsDenseMatGemv( P+(size_t)mvsize*nltgcr->_n, 'N', nltgcr->_n, vsize, 1.0, alph, 0.0, gg);
      }
      else
      {
         AatgsDenseMatGemv( V, 'T', nltgcr->_n, vidx+1, 1.0, r, 0.0, alph);
         AatgsDenseMatGemv( P, 'N', nltgcr->_n, vidx+1, 1.0, alph, 0.0, gg);
         AatgsDenseMatGemv( V+(size_t)(vsize+mvsize)*nltgcr->_n, 'T', nltgcr->_n, -mvsize, 1.0, r, 0.0, alph);
         AatgsDenseMatGemv( P+(size_t)(vsize+mvsize)*nltgcr->_n, 'N', nltgcr->_n, -mvsize, 1.0, alph, 1.0, gg);
      }

      tc = AatgsVecDdot(v, nltgcr->_n, gg);

      if(tc < 0.0)
      {
         printf("%10d | Warning: not descending, change directions\n", nltgcr->_nits);
         tc *= -1;
         AatgsVecScale(gg, nltgcr->_n, -1.0);
      }

      // line search
      int nls = 0;
      AATGS_DOUBLE lam = lam0;
      AATGS_MEMCPY(x2, x1, nltgcr->_n, AATGS_DOUBLE);
      AatgsVecAxpy( lam, gg, nltgcr->_n, x2);

      while(nls < nltgcr->_max_ls)
      {
         nls++;
         err = problem->_loss(vproblem, x2, nltgcr->_loss_history+nltgcr->_nits, r);nfe++;
         if(err != 0)
         {
            printf("%10d | Error: loss function failed\n", nltgcr->_nits);
            AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
            return -1;
         }
         if( nls >= nltgcr->_max_ls || nltgcr->_loss_history[nltgcr->_nits] < nltgcr->_loss_history[nltgcr->_nits] + nltgcr->_agc*tc*lam)
         {
            break;
         }
         lam *= 0.5;
         AatgsVecAxpy( -lam, gg, nltgcr->_n, x2);
      }
      if(nltgcr->_grad_history)
      {
         AATGS_MEMCPY(xgrad, r, nltgcr->_n, AATGS_DOUBLE);
      }
      AatgsVecScale( r, nltgcr->_n, -1.0);
      nltgcr->_grad_norm_history[nltgcr->_nits] = AatgsVecNorm2( r, nltgcr->_n);

      if(nls >= 2)
      {
         lam0 = 0.5 * lam;
      }
      else
      {
         AATGS_MIN( 2.0*lam, 1.0, lam0);
      }

      //printf("%10d | rT * gg = %e | line search steps = %d\n", nltgcr->_nits, tc, nls);

      if(nltgcr->_grad_norm_history[nltgcr->_nits] < nltgcr->_tol)
      {
         printf("%10d | Converge: gradient norm reached\n", nltgcr->_nits);
         AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
         return 1;
      }

      printf("%10d | %15.8e | %15.8e | %5d\n", nltgcr->_nits, nltgcr->_loss_history[nltgcr->_nits], nltgcr->_grad_norm_history[nltgcr->_nits], nfe);

      vidx++;
      vidx = vidx % (wsize+1);
      v = V + (size_t)vidx*nltgcr->_n;
      p = P + (size_t)vidx*nltgcr->_n;
      AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
      xrec[vidx] = 0.0;
      for(i = 0 ; i < nltgcr->_n ; i ++)
      {
         AATGS_MAX(xrec[vidx], fabs(r[i]), xrec[vidx]);
      }

      err = AatgsOptimizationNltgcrJVP( vproblem, nltgcr->_n, x2, 1.0, r,
                                 nltgcr->_grad_norm_history[nltgcr->_nits], -1.0, r, x1, v);nfe++;
      
      if(err != 0)
      {
         printf("%10d | Error: loss function failed\n", nltgcr->_nits);
         AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
         return -1;
      }

      // backup v
      AATGS_MEMCPY(x1, v, nltgcr->_n, AATGS_DOUBLE);
      
      flag = AatgsModifiedGSNltgcr( V, P, nltgcr->_n, wsize+1, vsize, vidx, &t, nltgcr->_safeguard, nltgcr->_restart_type, xrec, AATGS_EPS, 0.7071);
      
      restart++;
      if(nltgcr->_restart > 0 && restart > nltgcr->_restart - 1)
      {
         restart = 0;
         printf("%10d | Restarting: iteration count met\n", nltgcr->_nits);
         vsize = 1;
         AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
         AATGS_MEMCPY(v, x1, nltgcr->_n, AATGS_DOUBLE);
         t = 1.0 / AatgsVecNorm2(v, nltgcr->_n);
         AatgsVecScale(v, nltgcr->_n, t);
         AatgsVecScale(p, nltgcr->_n, t);
         xrec[vidx] = 0.0;
         for(i = 0 ; i < nltgcr->_n ; i ++)
         {
            AATGS_MAX(xrec[vidx], fabs(p[i]), xrec[vidx]);
         }
      }
      else if(flag == 0)
      {
         vsize ++;
         AATGS_MIN(vsize, wsize, vsize);
      }
      else if(flag == 1)
      {
         restart = 0;
         printf("%10d | Restarting: restart condition\n", nltgcr->_nits);
         vsize = 1;
         AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
         AATGS_MEMCPY(v, x1, nltgcr->_n, AATGS_DOUBLE);
         t = 1.0 / AatgsVecNorm2(v, nltgcr->_n);
         AatgsVecScale(v, nltgcr->_n, t);
         AatgsVecScale(p, nltgcr->_n, t);
         xrec[vidx] = 0.0;
         for(i = 0 ; i < nltgcr->_n ; i ++)
         {
            AATGS_MAX(xrec[vidx], fabs(p[i]), xrec[vidx]);
         }
      }
      else
      {
         printf("%10d | Error: linearly dependent basis\n", nltgcr->_nits);
         AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
         return -1;
      }

      if(nltgcr->_keep_x_history)
      {
         x1 += nltgcr->_n;
         x2 += nltgcr->_n;
      }
      else
      {
         x3 = x1;
         x1 = x2;
         x2 = x3;
      }

      if(nltgcr->_keep_grad_history)
      {
         xgrad += nltgcr->_n;
      }
   }// end of main loop

   printf("Maximum number of iterations reached with norm %e\n", nltgcr->_grad_norm_history[nltgcr->_nits-1]);
   AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);

   AATGS_FREE(V);
   AATGS_FREE(P);
   AATGS_FREE(r);
   AATGS_FREE(alph);
   AATGS_FREE(xrec);

   return 0;
}

/**
 * @brief   Linear version NLTGCR.
 * @details Linear version NLTGCR.
 * @note    Helper function.
 * @param[in]     voptimizer  Pointer to the optimizer.
 * @param[in]     vproblem    Pointer to the problem.
 * @param[in]     x           Point at which the loss function and its gradient are computed.
 * @param[out]    x_final     Final point.
 * @return  0 if maxits reached, 1 if tol reached, -1 if error occured.
 */
int AatgsOptimizationNltgcrRunLinear( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_nltgcr nltgcr = (paatgs_optimizer_nltgcr)optimizer->_optimizer_data;
   paatgs_problem problem = (paatgs_problem)vproblem;

   /* Initialize GD */
   AATGS_CALLOC(nltgcr->_loss_history, nltgcr->_maxits+1, AATGS_DOUBLE);
   if(nltgcr->_keep_x_history)
   {
      AATGS_CALLOC(nltgcr->_x_history, (size_t)(nltgcr->_maxits+1)*nltgcr->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(nltgcr->_x_history, 2*nltgcr->_n, AATGS_DOUBLE);
   }
   if(nltgcr->_keep_grad_history)
   {
      AATGS_CALLOC(nltgcr->_grad_history, (size_t)(nltgcr->_maxits+1)*nltgcr->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(nltgcr->_grad_history, nltgcr->_n, AATGS_DOUBLE);
   }
   AATGS_CALLOC(nltgcr->_grad_norm_history, nltgcr->_maxits+1, AATGS_DOUBLE);

   int wsize = nltgcr->_wsize;
   if(wsize <= 0)
   {
      wsize = nltgcr->_maxits;
   }
   AATGS_MIN(wsize, nltgcr->_maxits, wsize);

   if(wsize <= 0)
   {
      printf("Error: window size is invalid.\n");
      return -1;
   }

   AATGS_DOUBLE *P = NULL, *p = NULL;
   AATGS_DOUBLE *V = NULL, *v = NULL;
   AATGS_MALLOC(P, (size_t)(wsize+1)*nltgcr->_n, AATGS_DOUBLE);
   p = P;
   AATGS_MALLOC(V, (size_t)(wsize+1)*nltgcr->_n, AATGS_DOUBLE);
   v = V;

   AATGS_DOUBLE *x1 = nltgcr->_x_history;
   AATGS_DOUBLE *x2 = nltgcr->_x_history + nltgcr->_n;
   AATGS_DOUBLE *x3 = NULL;
   AATGS_DOUBLE *xgrad = nltgcr->_grad_history;

   AATGS_DOUBLE *r = NULL;
   AATGS_MALLOC(r, (size_t)5*nltgcr->_n, AATGS_DOUBLE);
   AATGS_DOUBLE *gg = r + nltgcr->_n;
   AATGS_DOUBLE *x0 = r + 2*nltgcr->_n;
   AATGS_DOUBLE *r0 = r + 3*nltgcr->_n;
   AATGS_DOUBLE *g0 = r + 4*nltgcr->_n;
   AATGS_DOUBLE nrmr, nrmr_tmp;
   
   AATGS_DOUBLE *alph = NULL;
   AATGS_MALLOC(alph, wsize+1, AATGS_DOUBLE);

   AATGS_MEMCPY(x1, x, nltgcr->_n, AATGS_DOUBLE);

   printf("Iteration  | Loss            | Grad norm       | NFE\n");

   AATGS_DOUBLE lam0 = nltgcr->_lr;
   AATGS_DOUBLE tc = 0.0;

   int i, err = 0;
   int vidx = 0, vsize = 0;
   AATGS_DOUBLE t, s;
   AATGS_DOUBLE *xrec = NULL;
   AATGS_MALLOC(xrec, wsize+1, AATGS_DOUBLE);
   int flag = 0;
   int nfe = 0;

   nltgcr->_nits = 0;
   
   err = problem->_loss(vproblem, x1, nltgcr->_loss_history+nltgcr->_nits, r);nfe++;
   if(err != 0)
   {
      printf("%10d | Error: loss function failed\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return -1;
   }
   nltgcr->_grad_norm_history[nltgcr->_nits] = AatgsVecNorm2( r, nltgcr->_n);

   if(nltgcr->_grad_history)
   {
      AATGS_MEMCPY(xgrad, r, nltgcr->_n, AATGS_DOUBLE);
   }
   AatgsVecScale( r, nltgcr->_n, -1.0);

   if(nltgcr->_grad_norm_history[nltgcr->_nits] < nltgcr->_tol)
   {
      printf("%10d | Converge: gradient norm reached\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return 1;
   }

   err = AatgsOptimizationNltgcrJVP( vproblem, nltgcr->_n, x1, 1.0, r, 
                                 nltgcr->_grad_norm_history[nltgcr->_nits], -1.0, r, x2, v);nfe++;

   if(err != 0)
   {
      printf("%10d | Error: loss function failed\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return -1;
   }

   t = AatgsVecNorm2(v, nltgcr->_n);
   if(t < AATGS_EPS)
   {
      printf("%10d | Warning: JVP is too small\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return -1;
   }

   t = 1.0/t;
   AATGS_MEMCPY( p, r, nltgcr->_n, AATGS_DOUBLE);
   AatgsVecScale(p, nltgcr->_n, t);
   AatgsVecScale(v, nltgcr->_n, t);

   // TODO: can save some time if do differently for different restart type and use OpenMP
   xrec[vidx] = 0.0;
   for(i = 0 ; i < nltgcr->_n ; i ++)
   {
      AATGS_MAX(xrec[vidx], fabs(r[i]), xrec[vidx]);
   }
   
   vsize ++;

   printf("%10d | %15.8e | %15.8e | %5d\n", nltgcr->_nits, nltgcr->_loss_history[nltgcr->_nits], nltgcr->_grad_norm_history[nltgcr->_nits], nfe);

   /* LOOP NLTGCR */
   int restart = 0;
   nltgcr->_nits++;
   // TODO: introducing i2 seems not necessary, remove it
   for(; nltgcr->_nits <= nltgcr->_maxits ; nltgcr->_nits++)
   {
      if(restart == 0)
      {
         AATGS_MEMCPY(x0, x1, nltgcr->_n, AATGS_DOUBLE);
         AATGS_MEMCPY(r0, r, nltgcr->_n, AATGS_DOUBLE);
      }

      int mvsize = vidx+1-vsize;
      if(mvsize >= 0)
      {
         AatgsDenseMatGemv( V+(size_t)mvsize*nltgcr->_n, 'T', nltgcr->_n, vsize, 1.0, r, 0.0, alph);
         AatgsDenseMatGemv( P+(size_t)mvsize*nltgcr->_n, 'N', nltgcr->_n, vsize, 1.0, alph, 0.0, gg);
         AatgsDenseMatGemv( V+(size_t)mvsize*nltgcr->_n, 'N', nltgcr->_n, vsize, 1.0, alph, 0.0, g0);
      }
      else
      {
         AatgsDenseMatGemv( V, 'T', nltgcr->_n, vidx+1, 1.0, r, 0.0, alph);
         AatgsDenseMatGemv( P, 'N', nltgcr->_n, vidx+1, 1.0, alph, 0.0, gg);
         AatgsDenseMatGemv( V, 'N', nltgcr->_n, vidx+1, 1.0, alph, 0.0, g0);
         AatgsDenseMatGemv( V+(size_t)(vsize+mvsize)*nltgcr->_n, 'T', nltgcr->_n, -mvsize, 1.0, r, 0.0, alph);
         AatgsDenseMatGemv( P+(size_t)(vsize+mvsize)*nltgcr->_n, 'N', nltgcr->_n, -mvsize, 1.0, alph, 1.0, gg);
         AatgsDenseMatGemv( V+(size_t)(vsize+mvsize)*nltgcr->_n, 'N', nltgcr->_n, -mvsize, 1.0, alph, 1.0, g0);
      }

      tc = AatgsVecDdot(v, nltgcr->_n, gg);

      if(tc < 0.0)
      {
         printf("%10d | Warning: not descending, change directions\n", nltgcr->_nits);
         tc *= -1;
         AatgsVecScale(gg, nltgcr->_n, -1.0);
      }

      // line search
      int nls = 0;
      AATGS_DOUBLE lam = lam0;
      nrmr = AatgsVecNorm2(r, nltgcr->_n); 
      nrmr_tmp = nrmr;
      AatgsVecAxpy( -lam, g0, nltgcr->_n, r);

      while(nls < nltgcr->_max_ls)
      {
         nls++;
         nrmr_tmp = nls != 0 ? AatgsVecNorm2(r, nltgcr->_n) : nrmr;
         if( nls >= nltgcr->_max_ls || nrmr_tmp < nrmr + nltgcr->_agc*tc*lam)
         {
            break;
         }
         lam *= 0.5;
         AatgsVecAxpy( lam, g0, nltgcr->_n, r);
      }

      /*
      if(nltgcr->_grad_history)
      {
         AATGS_MEMCPY(xgrad, r, nltgcr->_n, AATGS_DOUBLE);
      }
      AatgsVecScale( r, nltgcr->_n, -1.0);
      */

      AATGS_MEMCPY(x2, x1, nltgcr->_n, AATGS_DOUBLE);
      AatgsVecAxpy(lam, gg, nltgcr->_n, x2);

      nltgcr->_grad_norm_history[nltgcr->_nits] = nrmr_tmp;

      if(nls >= 2)
      {
         lam0 = 0.5 * lam;
      }
      else
      {
         AATGS_MIN( 2.0*lam, 1.0, lam0);
      }

      //printf("%10d | rT * gg = %e | line search steps = %d\n", nltgcr->_nits, tc, nls);

      if(nltgcr->_grad_norm_history[nltgcr->_nits] < nltgcr->_tol)
      {
         printf("%10d | Converge: gradient norm reached\n", nltgcr->_nits);
         AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
         return 1;
      }

      // TODO: This is not necessary, just for demonstrative purpose
      // so no nfe++
      err = problem->_loss(vproblem, x2, nltgcr->_loss_history+nltgcr->_nits, NULL);
      if(err != 0)
      {
         printf("%10d | Error: loss function failed\n", nltgcr->_nits);
         AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
         return -1;
      }

      printf("%10d | %15.8e | %15.8e | %5d\n", nltgcr->_nits, nltgcr->_loss_history[nltgcr->_nits], nltgcr->_grad_norm_history[nltgcr->_nits], nfe);

      vidx++;
      vidx = vidx % (wsize+1);
      v = V + (size_t)vidx*nltgcr->_n;
      p = P + (size_t)vidx*nltgcr->_n;
      AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
      xrec[vidx] = 0.0;
      for(i = 0 ; i < nltgcr->_n ; i ++)
      {
         AATGS_MAX(xrec[vidx], fabs(r[i]), xrec[vidx]);
      }

      err = AatgsOptimizationNltgcrJVP( vproblem, nltgcr->_n, x0, 1.0, r,
                                 nltgcr->_grad_norm_history[nltgcr->_nits], -1.0, r0, x1, v);nfe++;
      
      if(err != 0)
      {
         printf("%10d | Error: loss function failed\n", nltgcr->_nits);
         AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
         return -1;
      }

      flag = AatgsModifiedGSNltgcr( V, P, nltgcr->_n, wsize+1, vsize, vidx, &t, nltgcr->_safeguard, nltgcr->_restart_type, xrec, AATGS_EPS, 0.7071);
      
      restart++;
      // TODO: some copy paste here, definitly can be optimized
      if(nltgcr->_restart > 0 && restart > nltgcr->_restart - 1)
      {
         restart = 0;
         printf("%10d | Restarting: iteration count met\n", nltgcr->_nits);
         vsize = 1;

         err = problem->_loss(vproblem, x2, nltgcr->_loss_history+nltgcr->_nits, r);nfe++;
         if(err != 0)
         {
            printf("%10d | Error: loss function failed\n", nltgcr->_nits);
            AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
            return -1;
         }
         nltgcr->_grad_norm_history[nltgcr->_nits] = AatgsVecNorm2( r, nltgcr->_n);

         if(nltgcr->_grad_history)
         {
            AATGS_MEMCPY(xgrad, r, nltgcr->_n, AATGS_DOUBLE);
         }
         AatgsVecScale( r, nltgcr->_n, -1.0);

         if(nltgcr->_grad_norm_history[nltgcr->_nits] < nltgcr->_tol)
         {
            printf("%10d | Converge: gradient norm reached\n", nltgcr->_nits);
            AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
            return 1;
         }

         err = AatgsOptimizationNltgcrJVP( vproblem, nltgcr->_n, x2, 1.0, r, 
                                       nltgcr->_grad_norm_history[nltgcr->_nits], -1.0, r, x1, v);nfe++;

         if(err != 0)
         {
            printf("%10d | Error: loss function failed\n", nltgcr->_nits);
            AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
            return -1;
         }

         AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
         t = 1.0 / AatgsVecNorm2(v, nltgcr->_n);
         AatgsVecScale(v, nltgcr->_n, t);
         AatgsVecScale(p, nltgcr->_n, t);
         xrec[vidx] = 0.0;
         for(i = 0 ; i < nltgcr->_n ; i ++)
         {
            AATGS_MAX(xrec[vidx], fabs(p[i]), xrec[vidx]);
         }
      }
      else if(flag == 0)
      {
         vsize ++;
         AATGS_MIN(vsize, wsize, vsize);
      }
      else if(flag == 1)
      {
         restart = 0;
         printf("%10d | Restarting: restart condition\n", nltgcr->_nits);
         vsize = 1;

         err = problem->_loss(vproblem, x2, nltgcr->_loss_history+nltgcr->_nits, r);nfe++;
         if(err != 0)
         {
            printf("%10d | Error: loss function failed\n", nltgcr->_nits);
            AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
            return -1;
         }
         nltgcr->_grad_norm_history[nltgcr->_nits] = AatgsVecNorm2( r, nltgcr->_n);

         if(nltgcr->_grad_history)
         {
            AATGS_MEMCPY(xgrad, r, nltgcr->_n, AATGS_DOUBLE);
         }
         AatgsVecScale( r, nltgcr->_n, -1.0);

         if(nltgcr->_grad_norm_history[nltgcr->_nits] < nltgcr->_tol)
         {
            printf("%10d | Converge: gradient norm reached\n", nltgcr->_nits);
            AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
            return 1;
         }

         err = AatgsOptimizationNltgcrJVP( vproblem, nltgcr->_n, x2, 1.0, r, 
                                       nltgcr->_grad_norm_history[nltgcr->_nits], -1.0, r, x1, v);nfe++;

         if(err != 0)
         {
            printf("%10d | Error: loss function failed\n", nltgcr->_nits);
            AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
            return -1;
         }

         AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
         t = 1.0 / AatgsVecNorm2(v, nltgcr->_n);
         AatgsVecScale(v, nltgcr->_n, t);
         AatgsVecScale(p, nltgcr->_n, t);
         xrec[vidx] = 0.0;
         for(i = 0 ; i < nltgcr->_n ; i ++)
         {
            AATGS_MAX(xrec[vidx], fabs(p[i]), xrec[vidx]);
         }
      }
      else
      {
         printf("%10d | Error: linearly dependent basis\n", nltgcr->_nits);
         AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
         return -1;
      }

      if(nltgcr->_keep_x_history)
      {
         x1 += nltgcr->_n;
         x2 += nltgcr->_n;
      }
      else
      {
         x3 = x1;
         x1 = x2;
         x2 = x3;
      }

      if(nltgcr->_keep_grad_history)
      {
         xgrad += nltgcr->_n;
      }
   }// end of main loop

   printf("Maximum number of iterations reached with norm %e\n", nltgcr->_grad_norm_history[nltgcr->_nits-1]);
   AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);

   AATGS_FREE(V);
   AATGS_FREE(P);
   AATGS_FREE(r);
   AATGS_FREE(alph);
   AATGS_FREE(xrec);

   return 0;
}

/**
 * @brief   Adaptive version NLTGCR.
 * @details Adaptive version NLTGCR.
 * @note    Helper function.
 * @param[in]     voptimizer  Pointer to the optimizer.
 * @param[in]     vproblem    Pointer to the problem.
 * @param[in]     x           Point at which the loss function and its gradient are computed.
 * @param[out]    x_final     Final point.
 * @return  0 if maxits reached, 1 if tol reached, -1 if error occured.
 */
int AatgsOptimizationNltgcrRunAdaptive( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_nltgcr nltgcr = (paatgs_optimizer_nltgcr)optimizer->_optimizer_data;
   paatgs_problem problem = (paatgs_problem)vproblem;

   /* Initialize GD */
   AATGS_CALLOC(nltgcr->_loss_history, nltgcr->_maxits+1, AATGS_DOUBLE);
   if(nltgcr->_keep_x_history)
   {
      AATGS_CALLOC(nltgcr->_x_history, (size_t)(nltgcr->_maxits+1)*nltgcr->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(nltgcr->_x_history, 2*nltgcr->_n, AATGS_DOUBLE);
   }
   if(nltgcr->_keep_grad_history)
   {
      AATGS_CALLOC(nltgcr->_grad_history, (size_t)(nltgcr->_maxits+1)*nltgcr->_n, AATGS_DOUBLE);
   }
   else
   {
      AATGS_CALLOC(nltgcr->_grad_history, nltgcr->_n, AATGS_DOUBLE);
   }
   AATGS_CALLOC(nltgcr->_grad_norm_history, nltgcr->_maxits+1, AATGS_DOUBLE);

   int wsize = nltgcr->_wsize;
   if(wsize <= 0)
   {
      wsize = nltgcr->_maxits;
   }
   AATGS_MIN(wsize, nltgcr->_maxits, wsize);

   if(wsize <= 0)
   {
      printf("Error: window size is invalid.\n");
      return -1;
   }

   AATGS_DOUBLE *P = NULL, *p = NULL;
   AATGS_DOUBLE *V = NULL, *v = NULL;
   AATGS_MALLOC(P, (size_t)(wsize+1)*nltgcr->_n, AATGS_DOUBLE);
   p = P;
   AATGS_MALLOC(V, (size_t)(wsize+1)*nltgcr->_n, AATGS_DOUBLE);
   v = V;

   AATGS_DOUBLE *x1 = nltgcr->_x_history;
   AATGS_DOUBLE *x2 = nltgcr->_x_history + nltgcr->_n;
   AATGS_DOUBLE *x3 = NULL;
   AATGS_DOUBLE *xgrad = nltgcr->_grad_history;

   AATGS_DOUBLE *r = NULL;
   AATGS_MALLOC(r, (size_t)5*nltgcr->_n, AATGS_DOUBLE);
   AATGS_DOUBLE *gg = r + nltgcr->_n;
   AATGS_DOUBLE *x0 = r + 2*nltgcr->_n;
   AATGS_DOUBLE *r0 = r + 3*nltgcr->_n;
   AATGS_DOUBLE *g0 = r + 4*nltgcr->_n;
   AATGS_DOUBLE nrmr, nrmr_tmp;
   
   AATGS_DOUBLE *alph = NULL;
   AATGS_MALLOC(alph, wsize+1, AATGS_DOUBLE);

   AATGS_MEMCPY(x1, x, nltgcr->_n, AATGS_DOUBLE);

   printf("Iteration  | Loss            | Grad norm       | NFE\n");

   AATGS_DOUBLE lam0 = nltgcr->_lr;
   AATGS_DOUBLE tc = 0.0;

   int i, err = 0;
   int vidx = 0, vsize = 0;
   AATGS_DOUBLE t, s;
   AATGS_DOUBLE *xrec = NULL;
   AATGS_MALLOC(xrec, wsize+1, AATGS_DOUBLE);
   int flag = 0;
   int nfe = 0;

   nltgcr->_nits = 0;
   
   err = problem->_loss(vproblem, x1, nltgcr->_loss_history+nltgcr->_nits, r);nfe++;
   if(err != 0)
   {
      printf("%10d | Error: loss function failed\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return -1;
   }
   nltgcr->_grad_norm_history[nltgcr->_nits] = AatgsVecNorm2( r, nltgcr->_n);

   if(nltgcr->_grad_history)
   {
      AATGS_MEMCPY(xgrad, r, nltgcr->_n, AATGS_DOUBLE);
   }
   AatgsVecScale( r, nltgcr->_n, -1.0);

   if(nltgcr->_grad_norm_history[nltgcr->_nits] < nltgcr->_tol)
   {
      printf("%10d | Converge: gradient norm reached\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return 1;
   }

   err = AatgsOptimizationNltgcrJVP( vproblem, nltgcr->_n, x1, 1.0, r, 
                                 nltgcr->_grad_norm_history[nltgcr->_nits], -1.0, r, x2, v);nfe++;

   if(err != 0)
   {
      printf("%10d | Error: loss function failed\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return -1;
   }

   t = AatgsVecNorm2(v, nltgcr->_n);
   if(t < AATGS_EPS)
   {
      printf("%10d | Warning: JVP is too small\n", nltgcr->_nits);
      AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
      return -1;
   }

   t = 1.0/t;
   AATGS_MEMCPY( p, r, nltgcr->_n, AATGS_DOUBLE);
   AatgsVecScale(p, nltgcr->_n, t);
   AatgsVecScale(v, nltgcr->_n, t);

   // TODO: can save some time if do differently for different restart type and use OpenMP
   xrec[vidx] = 0.0;
   for(i = 0 ; i < nltgcr->_n ; i ++)
   {
      AATGS_MAX(xrec[vidx], fabs(r[i]), xrec[vidx]);
   }
   
   vsize ++;

   printf("%10d | %15.8e | %15.8e | %5d\n", nltgcr->_nits, nltgcr->_loss_history[nltgcr->_nits], nltgcr->_grad_norm_history[nltgcr->_nits], nfe);

   /* LOOP NLTGCR */
   int restart = 0; int run_linear = 0; int lin_step = 0;
   nltgcr->_nits++;
   for(; nltgcr->_nits <= nltgcr->_maxits ; nltgcr->_nits++)
   {
      // backup for linear
      if(run_linear && restart == 0)
      {
         AATGS_MEMCPY(x0, x1, nltgcr->_n, AATGS_DOUBLE);
         AATGS_MEMCPY(r0, r, nltgcr->_n, AATGS_DOUBLE);
      }

      int mvsize = vidx+1-vsize;
      if(mvsize >= 0)
      {
         AatgsDenseMatGemv( V+(size_t)mvsize*nltgcr->_n, 'T', nltgcr->_n, vsize, 1.0, r, 0.0, alph);
         AatgsDenseMatGemv( P+(size_t)mvsize*nltgcr->_n, 'N', nltgcr->_n, vsize, 1.0, alph, 0.0, gg);
         AatgsDenseMatGemv( V+(size_t)mvsize*nltgcr->_n, 'N', nltgcr->_n, vsize, 1.0, alph, 0.0, g0);
      }
      else
      {
         AatgsDenseMatGemv( V, 'T', nltgcr->_n, vidx+1, 1.0, r, 0.0, alph);
         AatgsDenseMatGemv( P, 'N', nltgcr->_n, vidx+1, 1.0, alph, 0.0, gg);
         AatgsDenseMatGemv( V, 'N', nltgcr->_n, vidx+1, 1.0, alph, 0.0, g0);
         AatgsDenseMatGemv( V+(size_t)(vsize+mvsize)*nltgcr->_n, 'T', nltgcr->_n, -mvsize, 1.0, r, 0.0, alph);
         AatgsDenseMatGemv( P+(size_t)(vsize+mvsize)*nltgcr->_n, 'N', nltgcr->_n, -mvsize, 1.0, alph, 1.0, gg);
         AatgsDenseMatGemv( V+(size_t)(vsize+mvsize)*nltgcr->_n, 'N', nltgcr->_n, -mvsize, 1.0, alph, 1.0, g0);
      }

      tc = AatgsVecDdot(v, nltgcr->_n, gg);

      if(tc < 0.0)
      {
         printf("%10d | Warning: not descending, change directions\n", nltgcr->_nits);
         tc *= -1;
         AatgsVecScale(gg, nltgcr->_n, -1.0);
      }

      // line search
      int nls = 0;
      AATGS_DOUBLE lam = lam0;

      if(run_linear)
      {

         nrmr = AatgsVecNorm2(r, nltgcr->_n); 
         nrmr_tmp = nrmr;
         AatgsVecAxpy( -lam, g0, nltgcr->_n, r);

         while(nls < nltgcr->_max_ls)
         {
            nls++;
            nrmr_tmp = nls != 0 ? AatgsVecNorm2(r, nltgcr->_n) : nrmr;
            if( nls >= nltgcr->_max_ls || nrmr_tmp < nrmr + nltgcr->_agc*tc*lam)
            {
               break;
            }
            lam *= 0.5;
            AatgsVecAxpy( lam, g0, nltgcr->_n, r);
         }

         /*
         if(nltgcr->_grad_history)
         {
            AATGS_MEMCPY(xgrad, r, nltgcr->_n, AATGS_DOUBLE);
         }
         AatgsVecScale( r, nltgcr->_n, -1.0);
         */

         AATGS_MEMCPY(x2, x1, nltgcr->_n, AATGS_DOUBLE);
         AatgsVecAxpy(lam, gg, nltgcr->_n, x2);

         nltgcr->_grad_norm_history[nltgcr->_nits] = nrmr_tmp;

      }
      else
      {
         AATGS_MEMCPY(x2, x1, nltgcr->_n, AATGS_DOUBLE);
         AATGS_MEMCPY(r0, r, nltgcr->_n, AATGS_DOUBLE);
         AatgsVecAxpy( lam, gg, nltgcr->_n, x2);

         while(nls < nltgcr->_max_ls)
         {
            nls++;
            err = problem->_loss(vproblem, x2, nltgcr->_loss_history+nltgcr->_nits, r);nfe++;
            if(err != 0)
            {
               printf("%10d | Error: loss function failed\n", nltgcr->_nits);
               AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
               return -1;
            }
            if( nls >= nltgcr->_max_ls || nltgcr->_loss_history[nltgcr->_nits] < nltgcr->_loss_history[nltgcr->_nits] + nltgcr->_agc*tc*lam)
            {
               break;
            }
            lam *= 0.5;
            AatgsVecAxpy( -lam, gg, nltgcr->_n, x2);
         }
         // update linear residual for later use
         AatgsVecAxpy( -lam, g0, nltgcr->_n, r0);
         if(nltgcr->_grad_history)
         {
            AATGS_MEMCPY(xgrad, r, nltgcr->_n, AATGS_DOUBLE);
         }
         AatgsVecScale( r, nltgcr->_n, -1.0);
         nltgcr->_grad_norm_history[nltgcr->_nits] = AatgsVecNorm2( r, nltgcr->_n);
      }

      if(nls >= 2)
      {
         lam0 = 0.5 * lam;
      }
      else
      {
         AATGS_MIN( 2.0*lam, 1.0, lam0);
      }

      //printf("%10d | rT * gg = %e | line search steps = %d\n", nltgcr->_nits, tc, nls);

      if(!run_linear)
      {
         // Nonlinear mode, check if we should switch to linear mode
         // TODO: we can make theta an extra parameter for input
         if( (AatgsVecDdot(r0, nltgcr->_n, r) / AatgsVecNorm2(r, nltgcr->_n) / AatgsVecNorm2(r0, nltgcr->_n)) > 0.99)
         {
            run_linear = 1;
            lin_step = nltgcr->_nits;
            printf("%10d | Switch to linear mode\n", nltgcr->_nits);
            // store the current residual and solution
            // TODO: we might want to restart for some other fancy applications 
            AATGS_MEMCPY(x0, x2, nltgcr->_n, AATGS_DOUBLE);
            AATGS_MEMCPY(r0, r, nltgcr->_n, AATGS_DOUBLE);
         }
      }

      if(nltgcr->_grad_norm_history[nltgcr->_nits] < nltgcr->_tol)
      {
         printf("%10d | Converge: gradient norm reached\n", nltgcr->_nits);
         AATGS_MEMCPY(x_final, x1, nltgcr->_n, AATGS_DOUBLE);
         return 1;
      }

      flag = 0;
      if(run_linear)
      {
         // TODO: we can make this "10" an extra parameter for input
         if(nltgcr->_nits - lin_step == 10)
         {
            // This is necessary for checking the restart condition, nfe++ is required
            // g0 is no longer needed in this loop so we use its memory
            err = problem->_loss(vproblem, x2, nltgcr->_loss_history+nltgcr->_nits, g0); nfe++;
            if(err != 0)
            {
               printf("%10d | Error: loss function failed\n", nltgcr->_nits);
               AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
               return -1;
            }
            // TODO: we can make theta an extra parameter for input
            // note that we have switch the direction of r for convenience
            if( (- AatgsVecDdot(g0, nltgcr->_n, r) / AatgsVecNorm2(r, nltgcr->_n) / AatgsVecNorm2(g0, nltgcr->_n)) < 0.99)
            {
               run_linear = 0;
               flag = 2;
            }
            else
            {
               lin_step = nltgcr->_nits;
            }
         }
         else
         {
            // TODO: This is not necessary, just for demonstrative purpose
            // might be redundency when switching to linear mode but not a big deal
            // so no nfe++
            err = problem->_loss(vproblem, x2, nltgcr->_loss_history+nltgcr->_nits, NULL);
            if(err != 0)
            {
               printf("%10d | Error: loss function failed\n", nltgcr->_nits);
               AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
               return -1;
            }
         }
      }

      printf("%10d | %15.8e | %15.8e | %5d\n", nltgcr->_nits, nltgcr->_loss_history[nltgcr->_nits], nltgcr->_grad_norm_history[nltgcr->_nits], nfe);

      vidx++;
      vidx = vidx % (wsize+1);
      v = V + (size_t)vidx*nltgcr->_n;
      p = P + (size_t)vidx*nltgcr->_n;
      AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
      xrec[vidx] = 0.0;
      for(i = 0 ; i < nltgcr->_n ; i ++)
      {
         AATGS_MAX(xrec[vidx], fabs(r[i]), xrec[vidx]);
      }
      
      if(run_linear)
      {
         err = AatgsOptimizationNltgcrJVP( vproblem, nltgcr->_n, x0, 1.0, r,
                                    nltgcr->_grad_norm_history[nltgcr->_nits], -1.0, r0, x1, v);nfe++; 
      }
      else
      {
         err = AatgsOptimizationNltgcrJVP( vproblem, nltgcr->_n, x2, 1.0, r,
                                    nltgcr->_grad_norm_history[nltgcr->_nits], -1.0, r, x1, v);nfe++; 
         // backup v
         AATGS_MEMCPY(x1, v, nltgcr->_n, AATGS_DOUBLE);
      }
      
      if(err != 0)
      {
         printf("%10d | Error: loss function failed\n", nltgcr->_nits);
         AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
         return -1;
      }

      if(flag == 0)
      {
         flag = AatgsModifiedGSNltgcr( V, P, nltgcr->_n, wsize+1, vsize, vidx, &t, nltgcr->_safeguard, nltgcr->_restart_type, xrec, AATGS_EPS, 0.7071);
      }
      
      restart++;
      // TODO: some copy paste here, definitly can be optimized
      if(nltgcr->_restart > 0 && restart > nltgcr->_restart - 1)
      {
         restart = 0;
         printf("%10d | Restarting: iteration count met\n", nltgcr->_nits);
         vsize = 1;
         if(run_linear)
         {

            err = problem->_loss(vproblem, x2, nltgcr->_loss_history+nltgcr->_nits, r);nfe++;
            if(err != 0)
            {
               printf("%10d | Error: loss function failed\n", nltgcr->_nits);
               AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
               return -1;
            }
            nltgcr->_grad_norm_history[nltgcr->_nits] = AatgsVecNorm2( r, nltgcr->_n);

            if(nltgcr->_grad_history)
            {
               AATGS_MEMCPY(xgrad, r, nltgcr->_n, AATGS_DOUBLE);
            }
            AatgsVecScale( r, nltgcr->_n, -1.0);

            if(nltgcr->_grad_norm_history[nltgcr->_nits] < nltgcr->_tol)
            {
               printf("%10d | Converge: gradient norm reached\n", nltgcr->_nits);
               AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
               return 1;
            }

            err = AatgsOptimizationNltgcrJVP( vproblem, nltgcr->_n, x2, 1.0, r, 
                                          nltgcr->_grad_norm_history[nltgcr->_nits], -1.0, r, x1, v);nfe++;

            if(err != 0)
            {
               printf("%10d | Error: loss function failed\n", nltgcr->_nits);
               AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
               return -1;
            }

            AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
            t = 1.0 / AatgsVecNorm2(v, nltgcr->_n);
            AatgsVecScale(v, nltgcr->_n, t);
            AatgsVecScale(p, nltgcr->_n, t);
            xrec[vidx] = 0.0;
            for(i = 0 ; i < nltgcr->_n ; i ++)
            {
               AATGS_MAX(xrec[vidx], fabs(p[i]), xrec[vidx]);
            }

         }
         else
         {
            AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
            AATGS_MEMCPY(v, x1, nltgcr->_n, AATGS_DOUBLE);
            t = 1.0 / AatgsVecNorm2(v, nltgcr->_n);
            AatgsVecScale(v, nltgcr->_n, t);
            AatgsVecScale(p, nltgcr->_n, t);
            xrec[vidx] = 0.0;
            for(i = 0 ; i < nltgcr->_n ; i ++)
            {
               AATGS_MAX(xrec[vidx], fabs(p[i]), xrec[vidx]);
            }
         }
      }
      else if(flag == 0)
      {
         vsize ++;
         AATGS_MIN(vsize, wsize, vsize);
      }
      else if(flag == 1)
      {
         restart = 0;
         printf("%10d | Restarting: restart condition\n", nltgcr->_nits);
         vsize = 1;
         if(run_linear)
         {
            err = problem->_loss(vproblem, x2, nltgcr->_loss_history+nltgcr->_nits, r);nfe++;
            if(err != 0)
            {
               printf("%10d | Error: loss function failed\n", nltgcr->_nits);
               AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
               return -1;
            }
            nltgcr->_grad_norm_history[nltgcr->_nits] = AatgsVecNorm2( r, nltgcr->_n);

            if(nltgcr->_grad_history)
            {
               AATGS_MEMCPY(xgrad, r, nltgcr->_n, AATGS_DOUBLE);
            }
            AatgsVecScale( r, nltgcr->_n, -1.0);

            if(nltgcr->_grad_norm_history[nltgcr->_nits] < nltgcr->_tol)
            {
               printf("%10d | Converge: gradient norm reached\n", nltgcr->_nits);
               AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
               return 1;
            }

            err = AatgsOptimizationNltgcrJVP( vproblem, nltgcr->_n, x2, 1.0, r, 
                                          nltgcr->_grad_norm_history[nltgcr->_nits], -1.0, r, x1, v);nfe++;

            if(err != 0)
            {
               printf("%10d | Error: loss function failed\n", nltgcr->_nits);
               AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
               return -1;
            }

            AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
            t = 1.0 / AatgsVecNorm2(v, nltgcr->_n);
            AatgsVecScale(v, nltgcr->_n, t);
            AatgsVecScale(p, nltgcr->_n, t);
            xrec[vidx] = 0.0;
            for(i = 0 ; i < nltgcr->_n ; i ++)
            {
               AATGS_MAX(xrec[vidx], fabs(p[i]), xrec[vidx]);
            }
         }
         else
         {
            AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
            AATGS_MEMCPY(v, x1, nltgcr->_n, AATGS_DOUBLE);
            t = 1.0 / AatgsVecNorm2(v, nltgcr->_n);
            AatgsVecScale(v, nltgcr->_n, t);
            AatgsVecScale(p, nltgcr->_n, t);
            xrec[vidx] = 0.0;
            for(i = 0 ; i < nltgcr->_n ; i ++)
            {
               AATGS_MAX(xrec[vidx], fabs(p[i]), xrec[vidx]);
            }
         }
      }
      else if(flag == 2)
      {
         restart = 0;
         printf("%10d | Restarting: switch back to nonlinear\n", nltgcr->_nits);
         vsize = 1;
         AATGS_MEMCPY(p, r, nltgcr->_n, AATGS_DOUBLE);
         AATGS_MEMCPY(v, x1, nltgcr->_n, AATGS_DOUBLE);
         t = 1.0 / AatgsVecNorm2(v, nltgcr->_n);
         AatgsVecScale(v, nltgcr->_n, t);
         AatgsVecScale(p, nltgcr->_n, t);
         xrec[vidx] = 0.0;
         for(i = 0 ; i < nltgcr->_n ; i ++)
         {
            AATGS_MAX(xrec[vidx], fabs(p[i]), xrec[vidx]);
         }
      }
      else
      {
         printf("%10d | Error: linearly dependent basis\n", nltgcr->_nits);
         AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);
         return -1;
      }

      if(nltgcr->_keep_x_history)
      {
         x1 += nltgcr->_n;
         x2 += nltgcr->_n;
      }
      else
      {
         x3 = x1;
         x1 = x2;
         x2 = x3;
      }

      if(nltgcr->_keep_grad_history)
      {
         xgrad += nltgcr->_n;
      }
   }// end of main loop

   printf("Maximum number of iterations reached with norm %e\n", nltgcr->_grad_norm_history[nltgcr->_nits-1]);
   AATGS_MEMCPY(x_final, x2, nltgcr->_n, AATGS_DOUBLE);

   AATGS_FREE(V);
   AATGS_FREE(P);
   AATGS_FREE(r);
   AATGS_FREE(alph);
   AATGS_FREE(xrec);

   return 0;
}

int AatgsOptimizationNltgcrRun( void *voptimizer, void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *x_final)
{
   /* set problem */
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_nltgcr nltgcr = (paatgs_optimizer_nltgcr)optimizer->_optimizer_data;
   paatgs_problem problem = (paatgs_problem)vproblem;
   nltgcr->_problem = problem;
   nltgcr->_n = problem->_n;

   if(nltgcr->_nits > 0)
   {
      printf("Error: cannot rerun optimization.\n");
      return -1;
   }

   switch(nltgcr->_type)
   {
      case AATGS_OPTIMIZER_NLTGCR_TYPE_NONLINEAR:
      {
         printf("============================================\n");
         printf("Running Nonliner NLTGCR\n");
         printf("Window size (history vector used in orthogonalization): %d\n", nltgcr->_wsize);
         printf("Learning rate (mu): %e\n", nltgcr->_lr);
         printf("Line search steps: (max_ls): %d\n", nltgcr->_max_ls);
         switch(nltgcr->_restart_type)
         {
            case AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NONE:
            {
               printf("Restart type: none\n");
               break;
            }
            case AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NORM_BOUND:
            {
               printf("Restart type: inf norm bound\n");
               printf("Restart safeguard: (safeguard): %e\n", nltgcr->_safeguard);
               break;
            }
            default:
            {
               printf("Restart type: none\n");
               break;
            }
         }
         printf("============================================\n");
         return AatgsOptimizationNltgcrRunNonlinear(voptimizer, vproblem, x, x_final);
         break;
      }
      case AATGS_OPTIMIZER_NLTGCR_TYPE_LINEAR:
      {
         printf("============================================\n");
         printf("Running Linear NLTGCR\n");
         printf("Window size (history vector used in orthogonalization): %d\n", nltgcr->_wsize);
         printf("Learning rate (mu): %e\n", nltgcr->_lr);
         printf("Line search steps: (max_ls): %d\n", nltgcr->_max_ls);
         switch(nltgcr->_restart_type)
         {
            case AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NONE:
            {
               printf("Restart type: none\n");
               break;
            }
            case AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NORM_BOUND:
            {
               printf("Restart type: inf norm bound\n");
               printf("Restart safeguard: (safeguard): %e\n", nltgcr->_safeguard);
               break;
            }
            default:
            {
               printf("Restart type: none\n");
               break;
            }
         }
         printf("============================================\n");
         return AatgsOptimizationNltgcrRunLinear(voptimizer, vproblem, x, x_final);
         break;
      }
      case AATGS_OPTIMIZER_NLTGCR_TYPE_ADAPTIVE:
      {
         printf("============================================\n");
         printf("Running Adaptive NLTGCR\n");
         printf("Window size (history vector used in orthogonalization): %d\n", nltgcr->_wsize);
         printf("Learning rate (mu): %e\n", nltgcr->_lr);
         printf("Line search steps: (max_ls): %d\n", nltgcr->_max_ls);
         switch(nltgcr->_restart_type)
         {
            case AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NONE:
            {
               printf("Restart type: none\n");
               break;
            }
            case AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NORM_BOUND:
            {
               printf("Restart type: inf norm bound\n");
               printf("Restart safeguard: (safeguard): %e\n", nltgcr->_safeguard);
               break;
            }
            default:
            {
               printf("Restart type: none\n");
               break;
            }
         }
         printf("============================================\n");
         return AatgsOptimizationNltgcrRunAdaptive(voptimizer, vproblem, x, x_final);
         break;
      }
      default:
      {
         printf("Error: unknown/unimplemented Anderson type, exit.\n");
         return -1;
      }
   }


   return 0; // maxits reached
}

int AatgsOptimizationNltgcrGetInfo( void *voptimizer, int *nitsp, AATGS_DOUBLE **xp, AATGS_DOUBLE **lossp, AATGS_DOUBLE **gradp, AATGS_DOUBLE **grad_normp)
{
   paatgs_optimizer optimizer = (paatgs_optimizer)voptimizer;
   paatgs_optimizer_nltgcr nltgcr = (paatgs_optimizer_nltgcr)optimizer->_optimizer_data;

   if(nitsp)
   {
      *nitsp = nltgcr->_nits;
   }

   if(xp)
   {
      *xp = nltgcr->_x_history;
   }

   if(lossp)
   {
      *lossp = nltgcr->_loss_history;
   }

   if(gradp)
   {
      *gradp = nltgcr->_grad_history;
   }

   if(grad_normp)
   {
      *grad_normp = nltgcr->_grad_norm_history;
   }

   return 0;
}