#include "lennardjones.h"
#include "../ops/vecops.h"
#include "../ops/matops.h"

void* AatgsProblemLennardJonesCreate( int nat, int dim)
{
   paatgs_problem_lennardjones lennardjones = NULL;
   AATGS_MALLOC(lennardjones, 1, aatgs_problem_lennardjones);

   lennardjones->_dim = dim;
   lennardjones->_nat = nat;

   lennardjones->_n = nat*dim;
   size_t dwork_size = (size_t)lennardjones->_n;
#ifdef AATGS_USING_OPENMP
   dwork_size += (size_t)lennardjones->_dim*omp_get_max_threads();
   dwork_size += (size_t)lennardjones->_n*omp_get_max_threads();
#else
   dwork_size += (size_t)lennardjones->_dim;
#endif
   AATGS_MALLOC(lennardjones->_dwork, dwork_size, AATGS_DOUBLE);

   paatgs_problem problem = NULL;
   AATGS_MALLOC(problem, 1, aatgs_problem);

   problem->_problem_data = (void*)lennardjones;
   problem->_n = lennardjones->_n;
   problem->_free_problem = &AatgsProblemLennardJonesFree;
   problem->_loss = &AatgsProblemLennardJonesLoss;
   problem->_hess = &AatgsProblemLennardJonesHess;
   
   return (void*)problem;

}

void AatgsProblemLennardJonesFree( void **vproblemp)
{
   if(*vproblemp)
   {
      paatgs_problem problem = (paatgs_problem)(*vproblemp);
      if(problem->_problem_data)
      {
         paatgs_problem_lennardjones lennardjones = (paatgs_problem_lennardjones)(problem->_problem_data);
         AATGS_FREE(lennardjones->_dwork);
         AATGS_FREE(lennardjones);
      }
      AATGS_FREE(*vproblemp);
   }
}

int AatgsProblemLennardJonesLoss( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *lossp, AATGS_DOUBLE *dloss)
{
   paatgs_problem problem = (paatgs_problem)vproblem;
   paatgs_problem_lennardjones lennardjones = (paatgs_problem_lennardjones)(problem->_problem_data);

   if(lennardjones->_dim != 3)
   {
      printf("Error: Lennard-Jones problem only supports 3D.\n");
      return -1;
   }

   AATGS_DOUBLE *y;
   AATGS_DOUBLE loss = 0.0;
   if(dloss)
   {
      // output is needed
      y = dloss;
      AatgsVecFill(y, lennardjones->_n, 0.0);
   }

#ifdef AATGS_USING_OPENMP
   if(!omp_in_parallel() && omp_get_max_threads() > 1)
   {
      #pragma omp parallel
      {
         int i, j, k;
         AATGS_DOUBLE *yi;
         AATGS_DOUBLE dd;
         AATGS_DOUBLE *dvec = lennardjones->_dwork + lennardjones->_n + omp_get_thread_num()*lennardjones->_dim;
         AATGS_DOUBLE loss_loc = 0.0;
         AATGS_DOUBLE *dloss_global = lennardjones->_dwork +
                                      lennardjones->_n +
                                      omp_get_max_threads()*lennardjones->_dim;
         AATGS_DOUBLE *dloss_loc = lennardjones->_dwork +
                                     lennardjones->_n +
                                     omp_get_max_threads()*lennardjones->_dim +
                                     omp_get_thread_num()*lennardjones->_n;
         AatgsVecFill(dloss_loc, lennardjones->_n, 0.0);
         #pragma omp for AATGS_DEFAULT_OPENMP_SCHEDULE
         for(i = 0 ; i < lennardjones->_nat ; i ++)
         {
            AATGS_DOUBLE *xi = x + (size_t)i*lennardjones->_dim;
            if(dloss)
            {
               yi = dloss_loc + (size_t)i*lennardjones->_dim;
            }
            for(j = 0 ; j < i ; j ++)
            {
               AATGS_DOUBLE *xj = x + (size_t)j*lennardjones->_dim;
               for(k = 0 ; k < lennardjones->_dim ; k ++)
               {
                  dvec[k] = xi[k] - xj[k];
               }
               dd = 1.0/AatgsVecDdot(dvec, lennardjones->_dim, dvec);
               // TODO: only for 3D, if want to generalize, need to rewrite this
               AATGS_DOUBLE r6 = dd*dd*dd;
               AATGS_DOUBLE r12 = r6*r6;
               if(lossp)
               {
                  loss_loc += 4.0*(r12 - r6);
               }
               if(dloss)
               {
                  AATGS_DOUBLE tt = -24.0*dd*(2.0*r12 - r6);
                  AATGS_DOUBLE *yj = dloss_loc + (size_t)j*lennardjones->_dim;
                  for(k = 0 ; k < lennardjones->_dim ; k ++)
                  {
                     yi[k] += tt*dvec[k];
                     yj[k] -= tt*dvec[k];
                  }
               }
            }
         }
         #pragma omp critical
         {
            loss += loss_loc;
         }
         if(dloss)
         {
            #pragma omp for AATGS_DEFAULT_OPENMP_SCHEDULE
            for(i = 0 ; i < lennardjones->_n ; i ++)
            {
               for(j = 0 ; j < omp_get_max_threads() ; j ++)
               {
                  y[i] += dloss_global[i + (size_t)j*lennardjones->_n];
               }
            }
         }
      }
   }
   else
   {
#endif
      int i, j, k;
      AATGS_DOUBLE *yi;
      AATGS_DOUBLE dd;
      AATGS_DOUBLE *dvec = lennardjones->_dwork + lennardjones->_n;
      for(i = 0 ; i < lennardjones->_nat ; i ++)
      {
         AATGS_DOUBLE *xi = x + (size_t)i*lennardjones->_dim;
         if(dloss)
         {
            yi = y + (size_t)i*lennardjones->_dim;
         }
         for(j = 0 ; j < i ; j ++)
         {
            AATGS_DOUBLE *xj = x + (size_t)j*lennardjones->_dim;
            for(k = 0 ; k < lennardjones->_dim ; k ++)
            {
               dvec[k] = xi[k] - xj[k];
            }
            dd = 1.0/AatgsVecDdot(dvec, lennardjones->_dim, dvec);
            // TODO: only for 3D, if want to generalize, need to rewrite this
            AATGS_DOUBLE r6 = dd*dd*dd;
            AATGS_DOUBLE r12 = r6*r6;
            if(lossp)
            {
               loss += 4.0*(r12 - r6);
            }
            if(dloss)
            {
               AATGS_DOUBLE tt = -24.0*dd*(2.0*r12 - r6);
               AATGS_DOUBLE *yj = y + (size_t)j*lennardjones->_dim;
               for(k = 0 ; k < lennardjones->_dim ; k ++)
               {
                  yi[k] += tt*dvec[k];
                  yj[k] -= tt*dvec[k];
               }
            }
         }
      }   
#ifdef AATGS_USING_OPENMP
   }
#endif

   if(lossp)
   {
      *lossp = loss;
   }

   return 0;
}

int AatgsProblemLennardJonesHess( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *hess)
{
   printf("Error: Hessian not implemented yet.\n");
   return -1;
}
