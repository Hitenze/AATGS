/**
 * @file test_mfem02.cpp
 * @brief   MFEM test example 02
 * @details This problem is from the MFEM example 02, see below for the original description.
 */

//                                MFEM Example 2
//
// Compile with: make test_mfem02.ex
//
// Sample runs:  test_mfem02.ex -m ./data/beam-tri.mesh
//               test_mfem02.ex -m ./data/beam-quad.mesh
//               test_mfem02.ex -m ./data/beam-tet.mesh
//               test_mfem02.ex -m ./data/beam-hex.mesh
//               test_mfem02.ex -m ./data/beam-wedge.mesh
//               test_mfem02.ex -m ./data/beam-quad.mesh -o 3 -sc
//               test_mfem02.ex -m ./data/beam-quad-nurbs.mesh
//               test_mfem02.ex -m ./data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder with f being a constant pull down
//               vector on boundary elements with attribute 2, and zero
//               otherwise. The geometry of the domain is assumed to be as
//               follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (pull down)
//
//               The example demonstrates the use of high-order and NURBS vector
//               finite element spaces with the linear elasticity bilinear form,
//               meshes with curved elements, and the definition of piece-wise
//               constant and vector coefficient objects. Static condensation is
//               also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "aatgs_headers.h"
#include "problem_mfem.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "./data/beam-tri.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
   int maxits = 200;
   double tol = 1e-12;
   double beta = 0.1;
   double safeguard = 1e03;
   int wsize = 4;
   int restart = -1;
   int tofile = 0;
   double lambda0 = 1.0;
   double mu0 = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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
   args.AddOption(&tofile, "-f", "--tofile",
                  "Write to file.");
   args.AddOption(&mu0, "-mu", "--mu",
                  "mu for the linear elasticity problem, between (0,infty), smaller means harder.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2)
   {
      cerr << "\nInput mesh should have at least two materials and "
           << "two boundary attributes! (See schematic in ex2.cpp)\n"
           << endl;
      return 3;
   }

   // 3. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh->NURBSext)
   {
      mesh->DegreeElevate(order, order);
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 5,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
   //    associated with the mesh nodes.
   FiniteElementCollection *fec;
   FiniteElementSpace *fespace;
   if (mesh->NURBSext)
   {
      fec = NULL;
      fespace = mesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      fespace = new FiniteElementSpace(mesh, fec, dim);
   }
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl << "Assembling: " << flush;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this case, b_i equals the boundary integral
   //    of f*phi_i where f represents a "pull down" force on the Neumann part
   //    of the boundary and phi_i are the basis functions in the finite element
   //    fespace. The force is defined by the VectorArrayCoefficient object f,
   //    which is a vector of Coefficient objects. The fact that f is non-zero
   //    on boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   LinearForm *b = new LinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   cout << "r.h.s. ... " << flush;
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.
   Vector lambda(mesh->attributes.Max());
   lambda = lambda0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh->attributes.Max());
   mu = mu0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,mu_func));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   cout << "matrix ... " << flush;
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "done." << endl;

   cout << "Size of linear system: " << A.Height() << endl;
   
   if(tofile)
   {
      MFEMWriteSparse2Csr( A, "A02.csr");
      MFEMWriteVector2Vec( B, "A02.b.vec");
      MFEMWriteVector2Vec( X, "A02.x.vec");
      std::ofstream out("A02.txt");
      A.Print(out);
      out.close();
   }

   // 11. Solve the system Ax=b with Anderson.

   int n = A.Size();

   /********************************
    * 11.1. Create the problem
    ********************************/

   void *problem = AatgsProblemMfemCreate( &A, n, &B);

   /********************************
    * 11.2. Create the optimizer
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
    * 11.3. Run the optimizer
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
    * 11.4. Get the results
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
    * 11.5. Free the optimizer
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

   // 12. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   // 13. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element. This allows us to save
   //     the displaced mesh as a curved mesh when using high-order finite
   //     element displacement field. We assume that the initial mesh (read from
   //     the file) is not higher order curved mesh compared to the chosen FE
   //     space.
   if (!mesh->NURBSext)
   {
      mesh->SetNodalFESpace(fespace);
   }

   // 14. Save the displaced mesh and the inverted solution (which gives the
   //     backward displacements to the original grid). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {
      GridFunction *nodes = mesh->GetNodes();
      *nodes += x;
      x *= -1;
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 15. Send the above data by socket to a GLVis server. Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 16. Free the used memory.
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete mesh;

   return 0;
}
