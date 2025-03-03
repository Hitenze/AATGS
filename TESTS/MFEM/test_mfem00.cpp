/**
 * @file test_mfem00.cpp
 * @brief   MFEM test example 00
 * @details This problem is from the MFEM example 00, see below for the original description.
 */

//                                MFEM Example 0
//
// Compile with: make test_mfem00.ex
//
// Sample runs:  test_mfem00.ex
//               test_mfem00.ex -m ./data/fichera.mesh
//               test_mfem00.ex -m ./data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "aatgs_headers.h"
#include "problem_mfem.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "./data/star.mesh";
   int order = 1;
   int maxits = 200;
   double tol = 1e-12;
   double beta = 0.1;
   double safeguard = 1e03;
   int wsize = 4;
   int restart = -1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&maxits, "-ai", "--maxits",
                  "Anderson max its.");
   args.AddOption(&tol, "-at", "--tol",
                  "Anderson tol.");
   args.AddOption(&beta, "-ab", "--beta",
                  "Anderson beta.");
   args.AddOption(&wsize, "-aw", "--wsize",
                  "Anderson window size.");
   args.AddOption(&restart, "-ar", "--restart",
                  "Anderson restart dimension.");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   mesh.UniformRefinement();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   // 8. Form the linear system A X = B. This includes eliminating boundary
   //    conditions, applying AMR constraints, and other transformations.
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 9. Solve the system Ax=b with Anderson.

   int n = A.Size();

   /********************************
    * 9.1. Create the problem
    ********************************/

   void *problem = AatgsProblemMfemCreate( &A, n, &B);

   /********************************
    * 9.2. Create the optimizer
    ********************************/

   void *gd = AatgsOptimizationGdCreate();
   AatgsOptimizationGdSetOptions(gd, maxits, tol);
   AatgsOptimizationGdSetParameters(gd, beta);

   void *anderson = AatgsOptimizationAndersonCreate();
   AatgsOptimizationGdSetOptions(anderson, maxits, tol);
   AatgsOptimizationAndersonSetParameters(anderson, 
                                          1.0, 
                                          beta, 
                                          wsize,
                                          restart,
                                          safeguard,
                                          AATGS_OPTIMIZER_ANDERSON_TYPE_AATGS, 
                                          AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NORM_BOUND);

   void *andersonr = AatgsOptimizationAndersonCreate();
   AatgsOptimizationAndersonSetOptions(andersonr, maxits, tol);
   AatgsOptimizationAndersonSetParameters(andersonr, 
                                          1.0, 
                                          beta, 
                                          wsize, 
                                          restart,
                                          safeguard,
                                          AATGS_OPTIMIZER_ANDERSON_TYPE_AARTGS, 
                                          AATGS_OPTIMIZER_ANDERSON_RESTART_TYPE_NORM_BOUND);

   void *nltgcr = AatgsOptimizationNltgcrCreate();
   AatgsOptimizationNltgcrSetOptions(nltgcr, maxits, tol);
   AatgsOptimizationNltgcrSetParameters(nltgcr, 
                                          1.0,
                                          10,
                                          wsize,
                                          0.001,
                                          0,
                                          restart,
                                          safeguard,
                                          AATGS_OPTIMIZER_NLTGCR_TYPE_NONLINEAR,
                                          AATGS_OPTIMIZER_NLTGCR_RESTART_TYPE_NORM_BOUND);

   /********************************
    * 9.3. Run the optimizer
    ********************************/

   double *x0 = (double*)malloc(n*sizeof(double));
   double *x_gd = (double*)malloc(n*sizeof(double));
   double *x_anderson = (double*)malloc(n*sizeof(double));
   double *x_andersonr = (double*)malloc(n*sizeof(double));
   double *x_nltgcr = (double*)malloc(n*sizeof(double));
   AATGS_MEMCPY(x0, X.GetData(), n, double);

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
   optimization = AatgsOptimizationCreate(nltgcr, problem);

   t1 = AatgsWtime();
   AatgsOptimizationRun(optimization, x0, x_nltgcr);
   t2 = AatgsWtime();
   printf("Total NLTCG optimization time: %f\n", t2-t1);

   AatgsOptimizationFree(&optimization);

   /********************************
    * 9.4. Get the results
    ********************************/

   int nits;
   double *x_history = NULL;
   double *loss_history = NULL;
   double *gradnorm_history = NULL;
   double *grad_history = NULL;

   AatgsOptimizationGdGetInfo(gd, &nits, &x_history, &loss_history, &grad_history, &grad_history);

   //printf("Number of iterations: %d\n", nits); 
   //AatgsTestPrintMatrix( x_history+nits*n-n, 1, n, 1);

   AATGS_MEMCPY( X.GetData(), x_anderson, n, double);

   /********************************
    * 9.5. Free the optimizer
    ********************************/

   // this will free both problem and optimizer
   AatgsOptimizationGdFree(&gd);
   AatgsOptimizationAndersonFree(&anderson);
   AatgsOptimizationAndersonFree(&andersonr);
   AatgsOptimizationNltgcrFree(&nltgcr);
   AatgsProblemMfemFree(&problem);
   free(x0);
   free(x_gd);
   free(x_anderson);
   free(x_andersonr);
   free(x_nltgcr);

   // 10. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   a.RecoverFEMSolution(X, b, x);
   x.Save("sol.gf");
   mesh.Save("mesh.mesh");

   return 0;
}
