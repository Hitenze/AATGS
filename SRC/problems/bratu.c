#include "bratu.h"
#include "../ops/vecops.h"
#include "../ops/matops.h"

void* AatgsProblemBratuCreate( AATGS_DOUBLE alpha, AATGS_DOUBLE lambda, int nx )
{
   paatgs_problem_bratu bratu = NULL;
   AATGS_MALLOC(bratu, 1, aatgs_problem_bratu);

   bratu->_nx = nx;
   bratu->_n = nx*nx;

   /* Next create the fd3d matrix, use a lazy implementation 
    * TODO: Carefully implement this
    */
   int nnz = 5*bratu->_n;

   AATGS_MALLOC(bratu->_A_i, bratu->_n+1, int);
   AATGS_MALLOC(bratu->_A_j, nnz, int);
   AATGS_MALLOC(bratu->_A_a, nnz, AATGS_DOUBLE);

   // loop to create data
   int i, j;
   bratu->_A_i[0] = 0;
   AATGS_DOUBLE diag = 4.0;
   AATGS_DOUBLE mone = -1.0;
   AATGS_DOUBLE h = 1.0 / (AATGS_DOUBLE)(nx+1);
   AATGS_DOUBLE ah = alpha*h/2.0;
   bratu->_lh = lambda*h*h;
   AATGS_DOUBLE diagu = -1.0 - ah;
   AATGS_DOUBLE diagl = -1.0 + ah;
   for(i = 0 ; i < nx ; i ++)
   {
      for(j = 0 ; j < nx ; j ++)
      {
         int idx = i*nx + j;
         if(i == 0)
         {
            // the first block
            if(j == 0)
            {
               bratu->_A_i[idx+1] = bratu->_A_i[idx];
               bratu->_A_j[bratu->_A_i[idx+1]] = idx;
               bratu->_A_a[bratu->_A_i[idx+1]++] = diag;
               if(idx + 1 < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagu;
               }
               if(idx + nx < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
            }
            else if(j == nx-1)
            {
               bratu->_A_i[idx+1] = bratu->_A_i[idx];
               if(idx - 1 >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagl;
               }
               bratu->_A_j[bratu->_A_i[idx+1]] = idx;
               bratu->_A_a[bratu->_A_i[idx+1]++] = diag;
               if(idx + nx < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
            }
            else
            {
               bratu->_A_i[idx+1] = bratu->_A_i[idx];
               if(idx - 1 >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagl;
               }
               bratu->_A_j[bratu->_A_i[idx+1]] = idx;
               bratu->_A_a[bratu->_A_i[idx+1]++] = diag;
               if(idx + 1 < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagu;
               }
               if(idx + nx < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
            }
         }
         else if(i == nx-1)
         {
            // the last block
            if(j == 0)
            {
               bratu->_A_i[idx+1] = bratu->_A_i[idx];
               if(idx - nx >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
               bratu->_A_j[bratu->_A_i[idx+1]] = idx;
               bratu->_A_a[bratu->_A_i[idx+1]++] = diag;
               if(idx + 1 < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagu;
               }
            }
            else if(j == nx-1)
            {
               bratu->_A_i[idx+1] = bratu->_A_i[idx];
               if(idx - nx >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
               if(idx - 1 >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagl;
               }
               bratu->_A_j[bratu->_A_i[idx+1]] = idx;
               bratu->_A_a[bratu->_A_i[idx+1]++] = diag;
            }
            else
            {
               bratu->_A_i[idx+1] = bratu->_A_i[idx];
               if(idx - nx >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
               if(idx - 1 >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagl;
               }
               bratu->_A_j[bratu->_A_i[idx+1]] = idx;
               bratu->_A_a[bratu->_A_i[idx+1]++] = diag;
               if(idx + 1 < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagu;
               }
            }
         }
         else
         {
            // the middle blocks
            if(j == 0)
            {
               bratu->_A_i[idx+1] = bratu->_A_i[idx];
               if(idx - nx >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
               bratu->_A_j[bratu->_A_i[idx+1]] = idx;
               bratu->_A_a[bratu->_A_i[idx+1]++] = diag;
               if(idx + 1 < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagu;
               }
               if(idx + nx < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
            }
            else if(j == nx-1)
            {
               bratu->_A_i[idx+1] = bratu->_A_i[idx];
               if(idx - nx >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
               if(idx - 1 >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagl;
               }
               bratu->_A_j[bratu->_A_i[idx+1]] = idx;
               bratu->_A_a[bratu->_A_i[idx+1]++] = diag;
               if(idx + nx < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
            }
            else
            {
               bratu->_A_i[idx+1] = bratu->_A_i[idx];
               if(idx - nx >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
               if(idx - 1 >= 0)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx - 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagl;
               }
               bratu->_A_j[bratu->_A_i[idx+1]] = idx;
               bratu->_A_a[bratu->_A_i[idx+1]++] = diag;
               if(idx + 1 < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + 1;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = diagu;
               }
               if(idx + nx < bratu->_n)
               {
                  bratu->_A_j[bratu->_A_i[idx+1]] = idx + nx;
                  bratu->_A_a[bratu->_A_i[idx+1]++] = mone;
               }
            }
         }
      }
   }

   // create working buffer
   AATGS_MALLOC(bratu->_dwork, bratu->_n, AATGS_DOUBLE);

   paatgs_problem problem = NULL;
   AATGS_MALLOC(problem, 1, aatgs_problem);

   problem->_problem_data = (void*)bratu;
   problem->_n = bratu->_n;
   problem->_free_problem = &AatgsProblemBratuFree;
   problem->_loss = &AatgsProblemBratuLoss;
   problem->_hess = &AatgsProblemBratuHess;
   
   return (void*)problem;

}

void AatgsProblemBratuFree( void **vproblemp)
{
   if(*vproblemp)
   {
      paatgs_problem problem = (paatgs_problem)(*vproblemp);
      if(problem->_problem_data)
      {
         paatgs_problem_bratu bratu = (paatgs_problem_bratu)(problem->_problem_data);
         AATGS_FREE(bratu->_A_i);
         AATGS_FREE(bratu->_A_j);
         AATGS_FREE(bratu->_A_a);
         AATGS_FREE(bratu->_dwork);
         AATGS_FREE(bratu);
      }
      AATGS_FREE(*vproblemp);
   }
}

int AatgsProblemBratuLoss( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *lossp, AATGS_DOUBLE *dloss)
{
   // loss = norm(grad(x))
   // grad = A*x = lh*exp(x)

   paatgs_problem problem = (paatgs_problem)vproblem;
   paatgs_problem_bratu bratu = (paatgs_problem_bratu)(problem->_problem_data);

   AATGS_DOUBLE *y;
   if(dloss)
   {
      // output is needed
      y = dloss;
   }
   else
   {
      y = bratu->_dwork;
   }

   AatgsCsrMv(bratu->_A_i, bratu->_A_j, bratu->_A_a, bratu->_n, bratu->_n, 'N', 1.0, x, 0.0, y);

   int i;
#ifdef AATGS_USING_OPENMP
   if(!omp_in_parallel())
   {
      #pragma omp parallel for private(i) AATGS_DEFAULT_OPENMP_SCHEDULE
      for(i = 0 ; i < bratu->_n ; i ++)
      {
         y[i] = y[i] - bratu->_lh*exp(x[i]);
      }
   }
   else
   {
#endif
      for(i = 0 ; i < bratu->_n ; i ++)
      {
         y[i] = y[i] - bratu->_lh*exp(x[i]);
      }
#ifdef AATGS_USING_OPENMP
   }
#endif

   if(lossp)
   {
      // output is needed
      *lossp = AatgsVecNorm2(y, bratu->_n);
   }

   return 0;
}

int AatgsProblemBratuHess( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *hess)
{
   // hess = A - lh*diag(exp(x))
   printf("Error: Hessian not implemented yet.\n");
   return -1;
}
