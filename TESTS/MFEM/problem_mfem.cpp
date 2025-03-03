#include "problem_mfem.hpp"

void* AatgsProblemMfemCreate( SparseMatrix *A, int n, Vector *b)
{
   paatgs_problem_mfem mfem = NULL;
   AATGS_MALLOC(mfem, 1, aatgs_problem_mfem);
   mfem->_n = n;
   mfem->_A = A;
   mfem->_b = b;
   AATGS_MALLOC(mfem->_dwork, mfem->_n, AATGS_DOUBLE);

   paatgs_problem problem = NULL;
   AATGS_MALLOC(problem, 1, aatgs_problem);

   problem->_problem_data = (void*)mfem;
   problem->_n = mfem->_n;
   problem->_free_problem = &AatgsProblemMfemFree;
   problem->_loss = &AatgsProblemMfemLoss;
   problem->_hess = &AatgsProblemMfemHess;
   
   return (void*)problem;

}

void AatgsProblemMfemFree( void **vproblemp)
{
   if(*vproblemp)
   {
      paatgs_problem problem = (paatgs_problem)(*vproblemp);
      if(problem->_problem_data)
      {
         paatgs_problem_mfem mfem = (paatgs_problem_mfem)(problem->_problem_data);
         AATGS_FREE(mfem->_dwork);
         AATGS_FREE(mfem);
      }
      AATGS_FREE(*vproblemp);
   }
}


void AatgsProblemMfemUpdate( void *vproblem, SparseMatrix *A, Vector *b)
{
   paatgs_problem problem = (paatgs_problem)(vproblem);
   paatgs_problem_mfem mfem = (paatgs_problem_mfem)(problem->_problem_data);
   mfem->_A = A;
   mfem->_b = b;
}

int AatgsProblemMfemLoss( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *lossp, AATGS_DOUBLE *dloss)
{
   paatgs_problem problem = (paatgs_problem)vproblem;
   paatgs_problem_mfem mfem = (paatgs_problem_mfem)(problem->_problem_data);

   AATGS_DOUBLE *y;
   if(dloss)
   {
      // output is needed
      y = dloss;
   }
   else
   {
      y = mfem->_dwork;
   }

   Vector vy(y, mfem->_n);
   Vector vx(x, mfem->_n);

   mfem->_A->Mult(vx, vy);
   subtract(vy, *mfem->_b, vy);

   if(lossp)
   {
      // output is needed
      *lossp = AatgsVecNorm2(y, mfem->_n);
   }

   return 0;
}

int AatgsProblemMfemHess( void *vproblem, AATGS_DOUBLE *x, AATGS_DOUBLE *hess)
{
   // hess = A - lh*diag(exp(x))
   printf("Error: Hessian not implemented yet.\n");
   return -1;
}


void MFEMWriteSparse2Csr( const mfem::SparseMatrix &matrix, const std::string &filename)
{
   std::ofstream file(filename);
   if (!file.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
   }

   int nrows = matrix.Height();
   int ncols = matrix.Width();
   int nnz = matrix.NumNonZeroElems();

   // Write nrows, ncols, nnz
   file << nrows << ", " << ncols << ", " << nnz << std::endl;

   // Write row pointers (I)
   const int *I = matrix.GetI();
   for (int i = 0; i <= nrows; ++i) {
      file << I[i] << (i < nrows ? ", " : "\n");
   }

   // Write column indices (J)
   const int *J = matrix.GetJ();
   for (int i = 0; i < nnz; ++i) {
      file << J[i] << (i < nnz - 1 ? ", " : "\n");
   }

   // Write values
   const double *data = matrix.GetData();
   for (int i = 0; i < nnz; ++i) {
      file << data[i] << (i < nnz - 1 ? ", " : "\n");
   }

   file.close();
}

void MFEMWriteVector2Vec(const mfem::Vector &vec, const std::string &filename) 
{
   std::ofstream file(filename);
   if (!file.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
   }

   // Write the size of the vector
   file << vec.Size() << std::endl;

   // Write the vector data, one element per line
   for (int i = 0; i < vec.Size(); ++i) {
      file << vec(i) << std::endl;
   }

   file.close();
}


void NLTGCRSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(oper != NULL, "the Operator is not set (use SetOperator).");

   // Quadrature points that are checked for negative Jacobians etc.
   Vector sk, rk, yk, rho, alpha;

   // r - r_{k+1}, c - descent direction
   sk.SetSize(width);    // x_{k+1}-x_k
   rk.SetSize(width);    // nabla(f(x_{k}))
   yk.SetSize(width);    // r_{k+1}-r_{k}
   rho.SetSize(m);       // 1/(dot(yk,sk)
   alpha.SetSize(m);     // rhok*sk'*c
   int last_saved_id = -1;

   int it;
   double norm0, norm, norm_goal;
   const bool have_b = (b.Size() == Height());

   if (!iterative_mode)
   {
      x = 0.0;
   }

   ProcessNewState(x);

   // r = F(x)-b
   oper->Mult(x, r);
   if (have_b) { r -= b; }

   c = r;           // initial descent direction

   norm0 = norm = initial_norm = Norm(r);
   if (print_options.first_and_last && !print_options.iterations)
   {
      mfem::out << "NLTGCR iteration " << setw(2) << 0
                << " : ||r|| = " << norm << "...\n";
   }
   norm_goal = std::max(rel_tol*norm, abs_tol);
   for (it = 0; true; it++)
   {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_options.iterations)
      {
         mfem::out << "NLTGCR iteration " <<  it
                   << " : ||r|| = " << norm;
         if (it > 0)
         {
            mfem::out << ", ||r||/||r_0|| = " << norm/norm0;
         }
         mfem::out << '\n';
      }

      if (norm <= norm_goal)
      {
         converged = true;
         break;
      }

      if (it >= max_iter)
      {
         converged = false;
         break;
      }

      rk = r;
      const double c_scale = ComputeScalingFactor(x, b);
      if (c_scale == 0.0)
      {
         converged = false;
         break;
      }
      add(x, -c_scale, c, x); // x_{k+1} = x_k - c_scale*c

      ProcessNewState(x);

      oper->Mult(x, r);
      if (have_b)
      {
         r -= b;
      }

      // NLTGCR - construct descent direction
      subtract(r, rk, yk);   // yk = r_{k+1} - r_{k}
      sk = c; sk *= -c_scale; //sk = x_{k+1} - x_{k} = -c_scale*c
      const double gamma = Dot(sk, yk)/Dot(yk, yk);

      // Save last m vectors
      last_saved_id = (last_saved_id == m-1) ? 0 : last_saved_id+1;
      *skArray[last_saved_id] = sk;
      *ykArray[last_saved_id] = yk;

      c = r;
      for (int i = last_saved_id; i > -1; i--)
      {
         rho(i) = 1.0/Dot((*skArray[i]),(*ykArray[i]));
         alpha(i) = rho(i)*Dot((*skArray[i]),c);
         add(c, -alpha(i), (*ykArray[i]), c);
      }
      if (it > m-1)
      {
         for (int i = m-1; i > last_saved_id; i--)
         {
            rho(i) = 1./Dot((*skArray[i]), (*ykArray[i]));
            alpha(i) = rho(i)*Dot((*skArray[i]),c);
            add(c, -alpha(i), (*ykArray[i]), c);
         }
      }

      c *= gamma;   // scale search direction
      if (it > m-1)
      {
         for (int i = last_saved_id+1; i < m ; i++)
         {
            double betai = rho(i)*Dot((*ykArray[i]), c);
            add(c, alpha(i)-betai, (*skArray[i]), c);
         }
      }
      for (int i = 0; i < last_saved_id+1 ; i++)
      {
         double betai = rho(i)*Dot((*ykArray[i]), c);
         add(c, alpha(i)-betai, (*skArray[i]), c);
      }

      norm = Norm(r);
   }

   final_iter = it;
   final_norm = norm;

   if (print_options.summary || (!converged && print_options.warnings) ||
       print_options.first_and_last)
   {
      mfem::out << "NLTGCR: Number of iterations: " << final_iter << '\n'
                << "   ||r|| = " << final_norm << '\n';
   }
   if (print_options.summary || (!converged && print_options.warnings))
   {
      mfem::out << "NLTGCR: No convergence!\n";
   }
}
