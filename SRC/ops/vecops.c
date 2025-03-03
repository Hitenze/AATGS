#include "vecops.h"

AATGS_DOUBLE AatgsVecNorm2(AATGS_DOUBLE *x, int n)
{
   int one = 1;
   return sqrt(AATGS_DDOT( &n, x, &one, x, &one));
}

AATGS_DOUBLE AatgsVecDdot(AATGS_DOUBLE *x, int n, AATGS_DOUBLE *y)
{
   int one = 1;
   return AATGS_DDOT( &n, x, &one, y, &one);
}

void AatgsVecRand(AATGS_DOUBLE *x, int n)
{
   int i;
   /* Note: no parallel implementation in order to make random vector consistant 
    * TODO: get some thread-safe random number generator
    */
   for(i = 0 ; i < n ; i ++)
   {
      x[i] = (AATGS_DOUBLE)rand() / (AATGS_DOUBLE)RAND_MAX;
   }
}

void AatgsVecRadamacher(AATGS_DOUBLE *x, int n)
{
   int i;
   /* Note: no parallel implementation in order to make random vector consistant
    * TODO: get some thread-safe random number generator
    */
   for(i = 0 ; i < n ; i ++)
   {
      x[i] = (AATGS_DOUBLE)rand() / (AATGS_DOUBLE)RAND_MAX;
      if(x[i] < 0.5)
      {
         x[i] = -1.0;
      }
      else
      {
         x[i] = 1.0;
      }
   }
}

void AatgsVecFill(AATGS_DOUBLE *x, int n, AATGS_DOUBLE val)
{
   int i;
#ifdef AATGS_USING_OPENMP
   if(!omp_in_parallel())
   {
      #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
#endif
      for(i = 0 ; i < n ; i ++)
      {
         x[i] = val;
      }
#ifdef AATGS_USING_OPENMP
   }
   else
   {
      for(i = 0 ; i < n ; i ++)
      {
         x[i] = val;
      }
   }
#endif
}

void AatgsVecScale(AATGS_DOUBLE *x, int n, AATGS_DOUBLE scale)
{
   int i;
   if(scale == 0.0)
   {
      AatgsVecFill( x, n, 0.0);
   }
   else
   {
#ifdef AATGS_USING_OPENMP
      if(!omp_in_parallel())
      {
         #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
#endif
         for(i = 0 ; i < n ; i ++)
         {
            x[i] *= scale;
         }
#ifdef AATGS_USING_OPENMP
      }
      else
      {
         for(i = 0 ; i < n ; i ++)
         {
            x[i] *= scale;
         }
      }
#endif
   }
}

void AatgsVecAxpy(AATGS_DOUBLE alpha, AATGS_DOUBLE *x, int n, AATGS_DOUBLE *y)
{
   int one = 1;
   AATGS_DAXPY(&n, &alpha, x, &one, y, &one);
}

void AatgsIVecFill(int *x, int n, int val)
{
   int i;
#ifdef AATGS_USING_OPENMP
   if(!omp_in_parallel())
   {
      #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
#endif
      for(i = 0 ; i < n ; i ++)
      {
         x[i] = val;
      }
#ifdef AATGS_USING_OPENMP
   }
   else
   {
      for(i = 0 ; i < n ; i ++)
      {
         x[i] = val;
      }
   }
#endif
}
