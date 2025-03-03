#include "matops.h"

int AatgsDenseMatGemv(void *data, char trans, int m, int n, AATGS_DOUBLE alpha, AATGS_DOUBLE *x, AATGS_DOUBLE beta, AATGS_DOUBLE *y)
{
   int one = 1;

   AATGS_DOUBLE *ddata = (AATGS_DOUBLE*) data;
   AATGS_DGEMV( &trans, &m, &n, &alpha, ddata, &m, x, &one, &beta, y, &one);

   return 0;
}

int AatgsModifiedGS( AATGS_DOUBLE *w, int n, int kdim, AATGS_DOUBLE *V, AATGS_DOUBLE *H, AATGS_DOUBLE *t, int k, AATGS_DOUBLE tol_orth, AATGS_DOUBLE tol_reorth)
{

   if(k < 0 || n == 0)
   {
      /* in this case, we don't have any previous vectors, return immediatly */
      return 0;
   }

   /*------------------------
    * 1: Modified Gram-Schmidt
    *------------------------*/

   int i;
   AATGS_DOUBLE t1, normw;

   AATGS_DOUBLE *v;

   /* compute initial ||w|| if we need to reorth */
   if(tol_reorth > 0.0)
   {
      normw = AatgsVecNorm2(w, n);
   }
   else
   {
      normw = 0.0;
   }

   for( i = 0 ; i <= k ; i ++)
   {
      /* inner produce and update H, w */
      v = V + i * n;

      t1 = AatgsVecDdot(w, n, v);
      H[k*(kdim+1)+i] = t1;

      AatgsVecAxpy( -t1, v, n, w);
   }

   /* Compute ||w|| */
   *t = AatgsVecNorm2( w, n);

   /*------------------------
    * 2: Re-orth step
    *------------------------*/

   /* t < tol_orth is considered be lucky breakdown */
   while( *t < normw * tol_reorth && *t >= tol_orth)
   {
      normw = *t;
      /* Re-orth */
      for (i = 0; i <= k; i++)
      {
         v = V + i * n;

         t1 = AatgsVecDdot(w, n, v);

         H[k*(kdim+1)+i] += t1;

         AatgsVecAxpy( -t1, v, n, w);
      }
      /* Compute ||w|| */
      *t = AatgsVecNorm2( w, n);

   }
   H[k*(kdim+1)+k+1] = *t;

   /* scale w in this function */
   AatgsVecScale( w, n, 1.0/(*t));

   return 0;

}

int AatgsCsrMv( int *ia, int *ja, AATGS_DOUBLE *aa, int nrows, int ncols, char trans, AATGS_DOUBLE alpha, AATGS_DOUBLE *x, AATGS_DOUBLE beta, AATGS_DOUBLE *y)
{
   
   int                  i, j, j1, j2;
   AATGS_DOUBLE         r, xi, *x_temp = NULL;
   AATGS_DOUBLE         one = 1.0;
   AATGS_DOUBLE         zero = 0.0;
#ifdef AATGS_USING_OPENMP
   int                  num_threads, my_thread_id;
   AATGS_DOUBLE         *y_temp;
#endif
   
   /* 1. Compute y = beta*y
      * note that when x==y and alpha != 0.0, we need to copy x
      */
   
   /* copy x when x==y, otherwise scale y would modify x 
      * TODO: memcpy or omp parallel?
      */
   if( (x == y) && (alpha != zero) )
   {
      if (trans == 'N') 
      {
         AATGS_MALLOC(x_temp, nrows, AATGS_DOUBLE);
         AATGS_MEMCPY(x_temp, y, nrows, AATGS_DOUBLE);
         x = x_temp;
      }
      else if(trans == 'T')
      {
         AATGS_MALLOC(x_temp, ncols, AATGS_DOUBLE);
         AATGS_MEMCPY(x_temp, y, ncols, AATGS_DOUBLE);
         x = x_temp;
      }
      else
      {
         return -1;
      }
   }
   
   /* now scale y */
   if(beta != one)
   {
      /* when beta == 1.0, y = y, do nothing */
      if(beta != zero)
      {
         /* y = beta*y */
         if (trans == 'N') 
         {
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for (i = 0; i < nrows; i++) 
               {
                  y[i] *= beta;
               }
            }
            else
            {
#endif
               for (i = 0; i < nrows; i++) 
               {
                  y[i] *= beta;
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
         }
         else if(trans == 'T')
         {
            /* if x == y need to create new x */
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for (i = 0; i < ncols; i++) 
               {
                  y[i] *= beta;
               }
            }
            else
            {
#endif
               for (i = 0; i < ncols; i++) 
               {
                  y[i] *= beta;
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
         }
         else
         {
            return -1;
         }
      }
      else
      {
         /* beta == 0.0, y = 0 */
         if (trans == 'N') 
         {
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for (i = 0; i < nrows; i++) 
               {
                  y[i] = zero;
               }
            }
            else
            {
#endif
               for (i = 0; i < nrows; i++) 
               {
                  y[i] = zero;
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
         }
         else if(trans == 'T')
         {
            /* if x == y need to create new x */
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
               for (i = 0; i < ncols; i++) 
               {
                  y[i] = zero;
               }
            }
            else
            {
#endif
               for (i = 0; i < ncols; i++) 
               {
                  y[i] = zero;
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
         }
         else
         {
            return -1;
         }
      }
   }
   
   
   /* 2. the matvec y = alpha*A*x + y
      * when alpha == 0 we have y = y, do nothing
      */
   
   if(alpha != zero)
   {
      if(alpha != one)
      {
         if (trans == 'N') 
         {
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i, j, r, j1, j2) AATGS_DEFAULT_OPENMP_SCHEDULE
               for (i = 0; i < nrows; i++) 
               {
                  r = zero;
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++) 
                  {
                     r += aa[j] * x[ja[j]];
                  }
                  y[i] += alpha*r;
               }
            }
            else
            {
#endif
               for (i = 0; i < nrows; i++) 
               {
                  r = 0.0;
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++) 
                  {
                     r += aa[j] * x[ja[j]];
                  }
                  y[i] += alpha*r;
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
         }
         else if(trans == 'T')
         {
#ifdef AATGS_USING_OPENMP
            /* create buffer for OpenMP when needed */
            num_threads = omp_get_max_threads();
            if(!omp_in_parallel() && num_threads>1)
            {
               AATGS_CALLOC(y_temp, num_threads * ncols, AATGS_DOUBLE);
               #pragma omp parallel private(i, j, j1, j2, my_thread_id, xi)
               {
                  my_thread_id = omp_get_thread_num();
                  AATGS_DOUBLE* y_local = y_temp + my_thread_id * ncols;
                  #pragma omp for AATGS_DEFAULT_OPENMP_SCHEDULE
                  /* sum to local buffer */
                  for (i = 0; i < nrows; i++) 
                  {
                     xi = alpha * x[i];
                     j1 = ia[i];
                     j2 = ia[i+1];
                     for (j = j1; j < j2; j++) 
                     {
                        y_local[ja[j]] += aa[j] * xi;
                     }
                  }
                  /* sumup the local buffer to y */
                  #pragma omp barrier
                  #pragma omp for AATGS_DEFAULT_OPENMP_SCHEDULE
                  for(i = 0 ; i < ncols ; i ++)
                  {
                     for(j = 0 ; j < num_threads ; j ++)
                     {
                        y[i] = y[i] + y_temp[i+j*ncols];
                     }
                  }
               }
               /* free temp buffer after openmp finished */
               AATGS_FREE(y_temp);
            }
            else
            {
#endif
               for (i = 0; i < nrows; i++) 
               {
                  xi = alpha * x[i];
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++) 
                  {
                     y[ja[j]] += aa[j] * xi;
                  }
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
         }
         else
         {
            return -1;
         }
      }
      else
      {
         if (trans == 'N') 
         {
#ifdef AATGS_USING_OPENMP
            if(!omp_in_parallel())
            {
               #pragma omp parallel for private(i, j, j1, j2) AATGS_DEFAULT_OPENMP_SCHEDULE
               for (i = 0; i < nrows; i++) 
               {
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++) 
                  {
                     y[i] += aa[j] * x[ja[j]];
                  }
               }
            }
            else
            {
#endif
               for (i = 0; i < nrows; i++) 
               {
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++) 
                  {
                     y[i] += aa[j] * x[ja[j]];
                  }
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
         }
         else if(trans == 'T')
         {
#ifdef AATGS_USING_OPENMP
            /* create buffer for OpenMP when needed */
            num_threads = omp_get_max_threads();
            if(!omp_in_parallel() && num_threads>1)
            {
               AATGS_CALLOC(y_temp, num_threads * ncols, AATGS_DOUBLE);
               #pragma omp parallel private(i, j, j1, j2, my_thread_id)
               {
                  my_thread_id = omp_get_thread_num();
                  AATGS_DOUBLE* y_local = y_temp + my_thread_id * ncols;
                  #pragma omp for AATGS_DEFAULT_OPENMP_SCHEDULE
                  /* sum to local buffer */
                  for (i = 0; i < nrows; i++) 
                  {
                     j1 = ia[i];
                     j2 = ia[i+1];
                     for (j = j1; j < j2; j++) 
                     {
                        y_local[ja[j]] += aa[j] * x[i];
                     }
                  }
                  /* sumup the local buffer to y */
                  #pragma omp barrier
                  #pragma omp for AATGS_DEFAULT_OPENMP_SCHEDULE
                  for(i = 0 ; i < ncols ; i ++)
                  {
                     for(j = 0 ; j < num_threads ; j ++)
                     {
                        y[i] = y[i] + y_temp[i+j*ncols];
                     }
                  }
               }
               /* free temp buffer after openmp finished */
               AATGS_FREE(y_temp);
            }
            else
            {
#endif
               for (i = 0; i < nrows; i++) 
               {
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++) 
                  {
                     y[ja[j]] += aa[j] * x[i];
                  }
               }
#ifdef AATGS_USING_OPENMP
            }
#endif
         }
         else
         {
            return -1;
         }
      }
   }
   
   if(x_temp)
   {
      AATGS_FREE( x_temp);
   }
   
   return 0;
}