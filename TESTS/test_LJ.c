/**
 * @brief   Test drive for the LJ problem
 * @details Test drive for the LJ problem
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>

#include "aatgs_headers.h"

AATGS_DOUBLE* ReadInitFromFile(const char* filename, int *n)
{
   int np, idx;
   AATGS_DOUBLE *x_init;
   
   FILE * fp;
   
   fp = fopen (filename, "r");
   
   if(fscanf(fp, "%d\n", &np) != 1)
   {
      printf("Error reading init file size\n");
      exit(1);
   }
   
   x_init = (AATGS_DOUBLE*)malloc(sizeof(AATGS_DOUBLE)*np);
   
   idx = 0;
   double io_buffer;
   for(int i = 0 ; i < np ; i ++)
   {
      if(fscanf(fp, "%lf", &io_buffer) != 1)
      {
         printf("Error reading init file at line %d\n", i+1);
         exit(1);
      }
      x_init[idx++] = (AATGS_DOUBLE)io_buffer;
   }
   
   fclose(fp);
   *n = np;
   
   return x_init;
}

int main(int argc, char **argv)
{
   int n;

   /********************************
    * 0. Read the input
    ********************************/
   
   int nat = 108;
   int dim = 3;
   int maxits = 200;
   AATGS_DOUBLE tol = 1e-14;
   AATGS_DOUBLE lr = 0.1;
   int wsize = 4;
   int restart = -1;
   AATGS_DOUBLE beta = 1.5e-04;
   AATGS_DOUBLE safeguard = 1e03;
   AATGS_DOUBLE mu = 1.0;
   int nthreads = 4;
   
   paatgs_io_handle io_handle = AatgsIoHandleCreate();

   AatgsIoHandleAddArg(io_handle, "nat", "num_atoms", "Number of atoms", AATGS_IO_TYPE_INT, &nat);
   AatgsIoHandleAddArg(io_handle, "dim", "dimension", "Problem dimension", AATGS_IO_TYPE_INT, &dim);
   AatgsIoHandleAddArg(io_handle, "its", "maxits", "Maximum number of iterations for anderson", AATGS_IO_TYPE_INT, &maxits);
   AatgsIoHandleAddArg(io_handle, "tol", "tolerance", "Tolerance for anderson", AATGS_IO_TYPE_DOUBLE, &tol);
   AatgsIoHandleAddArg(io_handle, "m", "window_size", "Window size for anderson", AATGS_IO_TYPE_INT, &wsize);
   AatgsIoHandleAddArg(io_handle, "kdim", "restart_dimension", "Restart dimension for anderson", AATGS_IO_TYPE_INT, &restart);
   AatgsIoHandleAddArg(io_handle, "lr", "learning_rate", "Learning rate for stand along fixed point iteration", AATGS_IO_TYPE_DOUBLE, &lr);
   AatgsIoHandleAddArg(io_handle, "beta", "beta", "Beta parameter for anderson", AATGS_IO_TYPE_DOUBLE, &beta);
   AatgsIoHandleAddArg(io_handle, "mu", "mu", "Mu parameter for anderson, learning rate for the fix point iteration in anderson", AATGS_IO_TYPE_DOUBLE, &mu);
   AatgsIoHandleAddArg(io_handle, "np", "nthreads", "Number of threads", AATGS_IO_TYPE_INT, &nthreads);

   AatgsIoHandlePhaseArgs(io_handle, argc, argv);

   AatgsIoHandlePringInfo(io_handle);
   AatgsIoHandleFree((void**)&io_handle);
   
#ifdef AATGS_USING_OPENMP
   if(nthreads > 1)
   {
      omp_set_num_threads(nthreads);
      printf("Using %d threads\n", nthreads);
   }
   else
   {
      omp_set_num_threads(1);
   }
#endif

   n = nat*dim;
   if(nat != 108 || dim != 3)
   {
      printf("Only support 108 atoms with 3 dimensions\n");
      exit(1);
   }

   /********************************
    * 1. Create the problem
    ********************************/

   void *problem = AatgsProblemLennardJonesCreate( nat, dim);

   /********************************
    * 2. Create the optimizer
    ********************************/

   void *gd = AatgsOptimizationGdCreate();
   AatgsOptimizationGdSetOptions(gd, maxits, tol);
   AatgsOptimizationGdSetParameters(gd, lr);

   void *anderson = AatgsOptimizationAndersonCreate();
   AatgsOptimizationGdSetOptions(anderson, maxits, tol);
   AatgsOptimizationAndersonSetParameters(anderson, 
                                          mu, 
                                          beta, 
                                          wsize,
                                          restart,
                                          safeguard,
                                          AATGS_OPTIMIZER_ANDERSON_TYPE_AATGS, 
                                          AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NORM_BOUND);

   void *andersonr = AatgsOptimizationAndersonCreate();
   AatgsOptimizationAndersonSetOptions(andersonr, maxits, tol);
   AatgsOptimizationAndersonSetParameters(andersonr, 
                                          mu, 
                                          beta, 
                                          wsize, 
                                          restart,
                                          safeguard,
                                          AATGS_OPTIMIZER_ANDERSON_TYPE_AARTGS, 
                                          AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NORM_BOUND);

   /********************************
    * 3. Run the optimizer
    ********************************/

   int n_init;
   AATGS_DOUBLE *x0 = ReadInitFromFile("LJ_init.dat", &n_init);
   assert(n_init == n);
   AATGS_DOUBLE *x_gd = (AATGS_DOUBLE*)malloc(n*sizeof(AATGS_DOUBLE));
   AATGS_DOUBLE *x_anderson = (AATGS_DOUBLE*)malloc(n*sizeof(AATGS_DOUBLE));
   AATGS_DOUBLE *x_andersonr = (AATGS_DOUBLE*)malloc(n*sizeof(AATGS_DOUBLE));

   void *optimization = AatgsOptimizationCreate(gd, problem);

   double t1, t2;
   t1 = AatgsWtime();
   AatgsOptimizationRun(optimization, x0, x_gd);
   t2 = AatgsWtime();
   printf("Total GD optimization time: %f\n", t2-t1);

   AatgsOptimizationFree(&optimization);
   optimization = AatgsOptimizationCreate(anderson, problem);

   t1 = AatgsWtime();
   AatgsOptimizationRun(optimization, x0, x_anderson);
   t2 = AatgsWtime();
   printf("Total Anderson optimization time: %f\n", t2-t1);

   AatgsOptimizationFree(&optimization);
   optimization = AatgsOptimizationCreate(andersonr, problem);

   t1 = AatgsWtime();
   AatgsOptimizationRun(optimization, x0, x_andersonr);
   t2 = AatgsWtime();
   printf("Total Anderson Reverse optimization time: %f\n", t2-t1);

   AatgsOptimizationFree(&optimization);

   /********************************
    * 4. Get the results
    ********************************/

   int nits;
   AATGS_DOUBLE *x_history = NULL;
   AATGS_DOUBLE *loss_history = NULL;
   AATGS_DOUBLE *gradnorm_history = NULL;
   AATGS_DOUBLE *grad_history = NULL;

   AatgsOptimizationGdGetInfo(gd, &nits, &x_history, &loss_history, &grad_history, &grad_history);

   //printf("Number of iterations: %d\n", nits); 
   //AatgsTestPrintMatrix( x_history+nits*n-n, 1, n, 1);

   /********************************
    * 5. Free the optimizer
    ********************************/

   // this will free both problem and optimizer
   AatgsOptimizationGdFree(&gd);
   AatgsOptimizationAndersonFree(&anderson);
   AatgsOptimizationAndersonFree(&andersonr);
   AatgsProblemLennardJonesFree(&problem);
   free(x0);
   free(x_gd);
   free(x_anderson);
   free(x_andersonr);
   
   return 0;
}