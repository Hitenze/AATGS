#ifndef AATGS_MFEM_HPP
#define AATGS_MFEM_HPP

/**
 * @file problem_mfem.hpp
 * @brief   MFEM optimization problems
 * @details MFEM optimization problems
 */

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "aatgs_headers.h"

using namespace std;
using namespace mfem;

typedef struct AATGS_PROBLEM_MFEM_STRUCT
{
   int            _n;

   SparseMatrix   *_A;
   Vector         *_b;

   AATGS_DOUBLE   *_dwork;
}aatgs_problem_mfem, *paatgs_problem_mfem;

void* AatgsProblemMfemCreate( SparseMatrix *A, int n, Vector *b);
void AatgsProblemMfemFree( void **vproblemp);
void AatgsProblemMfemUpdate( void *vproblem, SparseMatrix *A, Vector *b);
int AatgsProblemMfemLoss( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *lossp, AATGS_DOUBLE *dloss);
int AatgsProblemMfemHess( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *hess);

void MFEMWriteSparse2Csr( const mfem::SparseMatrix &matrix, const std::string &filename);
void MFEMWriteVector2Vec(const mfem::Vector &vec, const std::string &filename);

/** NLTGCR method for solving F(x)=b for a given operator F, by minimizing
    the norm of F(x) - b. Requires only the action of the operator F. */
class NLTGCRSolver : public NewtonSolver
{
protected:
   int m = 10;
   mutable Array<Vector *> skArray, ykArray;

   void DeleteStorageVectors()
   {
      for (int i = 0; i < skArray.Size(); i++)
      {
         delete skArray[i];
         delete ykArray[i];
      }
   }

   void InitializeStorageVectors()
   {
      DeleteStorageVectors();
      skArray.SetSize(m);
      ykArray.SetSize(m);
      for (int i = 0; i < m; i++)
      {
         skArray[i] = new Vector(width);
         ykArray[i] = new Vector(width);
         skArray[i]->UseDevice(true);
         ykArray[i]->UseDevice(true);
      }
   }

public:

   void* _problem;
   void* _mfem;

   NLTGCRSolver() : NewtonSolver() { }

#ifdef MFEM_USE_MPI
   NLTGCRSolver(MPI_Comm comm_) : NewtonSolver(comm_) { }
#endif

   virtual void SetOperator(const Operator &op)
   {
      NewtonSolver::SetOperator(op);
      InitializeStorageVectors();
   }

   void SetHistorySize(int dim)
   {
      m = dim;
      InitializeStorageVectors();
   }

   /// Solve the nonlinear system with right-hand side @a b.
   /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
   virtual void Mult(const Vector &b, Vector &x) const;

   virtual void SetPreconditioner(Solver &pr)
   { MFEM_WARNING("NLTGCR won't use the given preconditioner."); }
   virtual void SetSolver(Solver &solver)
   { MFEM_WARNING("NLTGCR won't use the given solver."); }

   virtual ~NLTGCRSolver() { DeleteStorageVectors(); }
};

#endif