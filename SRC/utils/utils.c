#include "utils.h"
#include "memory.h"

double AatgsWtime()
{
   struct timeval ctime;
   gettimeofday(&ctime, NULL);
   return (double)ctime.tv_sec + (double)0.000001 * ctime.tv_usec;
}

void *AatgsIoHandleCreate()
{
   paatgs_io_handle io_handle = NULL;
   AATGS_MALLOC(io_handle, 1, aatgs_io_handle);

   io_handle->_maxargs = AATGS_ARRAY_DEFAULT_SIZE;
   io_handle->_nargs = 0;
   io_handle->_arg_abbrev = NULL;
   AATGS_MALLOC(io_handle->_arg_abbrev, (size_t)io_handle->_maxargs, char *);
   io_handle->_arg_full = NULL;
   AATGS_MALLOC(io_handle->_arg_full, (size_t)io_handle->_maxargs, char *);
   io_handle->_arg_help = NULL;
   AATGS_MALLOC(io_handle->_arg_help, (size_t)io_handle->_maxargs, char *);
   io_handle->_arg_type = NULL;
   AATGS_MALLOC(io_handle->_arg_type, (size_t)io_handle->_maxargs, aatgs_io_type);
   io_handle->_arg = NULL;
   AATGS_MALLOC(io_handle->_arg, (size_t)io_handle->_maxargs, void*);

   return (void *)io_handle;
}

void AatgsIoHandleFree(void **viohandle)
{
   if(*viohandle)
   {
      paatgs_io_handle io_handle = (paatgs_io_handle)(*viohandle);
      int i;
      for(i = 0; i < io_handle->_nargs; i++)
      {
         AATGS_FREE(io_handle->_arg_abbrev[i]);
         AATGS_FREE(io_handle->_arg_full[i]);
         AATGS_FREE(io_handle->_arg_help[i]);
      }
      AATGS_FREE(io_handle->_arg_abbrev);
      AATGS_FREE(io_handle->_arg_full);
      AATGS_FREE(io_handle->_arg_help);
      AATGS_FREE(io_handle);
   }
}

void AatgsIoHandleAddArg(void *viohandle, char *arg_abbrev, char *arg_full, char *arg_help, aatgs_io_type arg_type, void *arg)
{
   paatgs_io_handle io_handle = (paatgs_io_handle)viohandle;
   if(io_handle->_nargs == io_handle->_maxargs)
   {
      int new_maxargs = (int)((AARGS_ARRAY_EXPAND_FACTOR) * (AATGS_DOUBLE)(io_handle->_maxargs));
      while(new_maxargs <= io_handle->_nargs)
      {
         new_maxargs = (int)((AARGS_ARRAY_EXPAND_FACTOR) * (AATGS_DOUBLE)(new_maxargs));
      }

      char **arg_abbrev;
      char **arg_full;
      char **arg_help;
      aatgs_io_type *arg_type;
      void **arg;

      AATGS_MALLOC(arg_abbrev, (size_t)new_maxargs, char *);
      AATGS_MALLOC(arg_full, (size_t)new_maxargs, char *);
      AATGS_MALLOC(arg_help, (size_t)new_maxargs, char *);
      AATGS_MALLOC(arg_type, (size_t)new_maxargs, aatgs_io_type);
      AATGS_MALLOC(arg, (size_t)new_maxargs, void*);

      int i;
      for(i = 0; i < io_handle->_nargs; i++)
      {
         AATGS_MALLOC(arg_abbrev[i], AATGS_IO_MAX_STRING_LENGTH, char);
         AATGS_MALLOC(arg_full[i], AATGS_IO_MAX_STRING_LENGTH, char);
         AATGS_MALLOC(arg_help[i], AATGS_IO_MAX_STRING_LENGTH, char);
         strcpy(arg_abbrev[i], io_handle->_arg_abbrev[i]);
         strcpy(arg_full[i], io_handle->_arg_full[i]);
         strcpy(arg_help[i], io_handle->_arg_help[i]);
         arg_type[i] = io_handle->_arg_type[i];
         arg[i] = io_handle->_arg[i];
         AATGS_FREE(io_handle->_arg_abbrev[i]);
         AATGS_FREE(io_handle->_arg_full[i]);
         AATGS_FREE(io_handle->_arg_help[i]);
      }

      AATGS_FREE(io_handle->_arg_abbrev);
      AATGS_FREE(io_handle->_arg_full);
      AATGS_FREE(io_handle->_arg_help);
      AATGS_FREE(io_handle->_arg_type);
      AATGS_FREE(io_handle->_arg);

      io_handle->_arg_abbrev = arg_abbrev;
      io_handle->_arg_full = arg_full;
      io_handle->_arg_help = arg_help;
      io_handle->_arg_type = arg_type;
      io_handle->_arg = arg;

      io_handle->_maxargs = new_maxargs;
   }

   AATGS_MALLOC(io_handle->_arg_abbrev[io_handle->_nargs], AATGS_IO_MAX_STRING_LENGTH, char);
   AATGS_MALLOC(io_handle->_arg_full[io_handle->_nargs], AATGS_IO_MAX_STRING_LENGTH, char);
   AATGS_MALLOC(io_handle->_arg_help[io_handle->_nargs], AATGS_IO_MAX_STRING_LENGTH, char);
   strcpy(io_handle->_arg_abbrev[io_handle->_nargs], arg_abbrev);
   strcpy(io_handle->_arg_full[io_handle->_nargs], arg_full);
   strcpy(io_handle->_arg_help[io_handle->_nargs], arg_help);
   io_handle->_arg_type[io_handle->_nargs] = arg_type;
   io_handle->_arg[io_handle->_nargs] = arg;

   io_handle->_nargs++;
}

void AatgsIoHandlePrintHelp(void *viohandle)
{
   paatgs_io_handle io_handle = (paatgs_io_handle)viohandle;
   int i;
   printf("========================================\n");
   printf("Arguments List:\n");
   printf("-help, --help: Print this help message\n");
   for(i = 0; i < io_handle->_nargs; i++)
   {
      printf("-%s, --%s: %s\n", io_handle->_arg_abbrev[i], io_handle->_arg_full[i], io_handle->_arg_help[i]);
   }
   printf("========================================\n");
}

void AatgsIoHandlePringInfo(void *viohandle)
{
   paatgs_io_handle io_handle = (paatgs_io_handle)viohandle;
   int i;
   printf("========================================\n");
   printf("Arguments Values:\n");
   for(i = 0; i < io_handle->_nargs; i++)
   {
      if(io_handle->_arg_type[i] == AATGS_IO_TYPE_INT)
      {
         printf("--%s: %d; %s\n", io_handle->_arg_full[i], *((int*)io_handle->_arg[i]), io_handle->_arg_help[i]);
      }
      else if(io_handle->_arg_type[i] == AATGS_IO_TYPE_DOUBLE)
      {
         printf("--%s: %g; %s\n", io_handle->_arg_full[i], *((AATGS_DOUBLE*)io_handle->_arg[i]), io_handle->_arg_help[i]);
      }
   }
   printf("========================================\n");
}

void AatgsIoHandlePhaseArgs(void *viohandle, int argc, char **argv)
{
   paatgs_io_handle io_handle = (paatgs_io_handle)viohandle;
   int i, j;
   for(i = 1; i < argc; i++)
   {
      if(argv[i][0] == '-')
      {
         if(argv[i][1] == '-')
         {
            // full name
            for(j = 0; j < io_handle->_nargs; j++)
            {
               if(strcmp(argv[i]+2, io_handle->_arg_full[j]) == 0)
               {
                  if(io_handle->_arg_type[j] == AATGS_IO_TYPE_INT)
                  {
                     *((int*)io_handle->_arg[j]) = atoi(argv[i+1]);
                  }
                  else if(io_handle->_arg_type[j] == AATGS_IO_TYPE_DOUBLE)
                  {
                     *((AATGS_DOUBLE*)io_handle->_arg[j]) = atof(argv[i+1]);
                  }
                  break;
               }
               // check for --help
               if(strcmp(argv[i]+2, "help") == 0)
               {
                  AatgsIoHandlePrintHelp(io_handle);
                  exit(0);
               }
            }
         }
         else
         {
            // abbrev name
            for(j = 0; j < io_handle->_nargs; j++)
            {
               if(strcmp(argv[i]+1, io_handle->_arg_abbrev[j]) == 0)
               {
                  if(io_handle->_arg_type[j] == AATGS_IO_TYPE_INT)
                  {
                     *((int*)io_handle->_arg[j]) = atoi(argv[i+1]);
                  }
                  else if(io_handle->_arg_type[j] == AATGS_IO_TYPE_DOUBLE)
                  {
                     *((AATGS_DOUBLE*)io_handle->_arg[j]) = atof(argv[i+1]);
                  }
                  break;
               }
               // check for -help
               if(strcmp(argv[i]+1, "help") == 0)
               {
                  AatgsIoHandlePrintHelp(io_handle);
                  exit(0);
               }
            }
         }
      }
   }
}

void *AatgsMatrixUniformRandom(int n, int d)
{
   int i, j, idx;
   AATGS_DOUBLE scale, *data;

   AATGS_MALLOC(data, (size_t)n * d, AATGS_DOUBLE);

   scale = pow(n, 1.0 / d);

   idx = 0;
   for (i = 0; i < n; i++)
   {
      for (j = 0; j < d; j++)
      {
         data[j * n + idx] = scale * (AATGS_DOUBLE)rand() / (AATGS_DOUBLE)RAND_MAX;
      }
      idx++;
   }

   return (void *)data;
}

void AatgsTestPrintMatrix(AATGS_DOUBLE *matrix, int m, int n, int ldim)
{
   int i, j;
   for (i = 0; i < m; i++)
   {
      for (j = 0; j < n; j++)
      {
         printf("%24.20f ", matrix[i + j * ldim]);
      }
      printf("\n");
   }
}

void AatgsTestPrintMatrixToFile(FILE *file, AATGS_DOUBLE *matrix, int m, int n, int ldim)
{
   int i, j;
   for (i = 0; i < m; i++)
   {
      for (j = 0; j < n; j++)
      {
         fprintf(file, "%16.6f ", matrix[i + j * ldim]);
      }
      fprintf(file, "\n");
   }
}

void AatgsTestPrintCSRMatrixPattern(int *A_i, int *A_j, int m, int n)
{
   int *temp_matrix = NULL;
   AATGS_CALLOC(temp_matrix, (size_t)m * n, int);

   int i, j1, j2, j;
   for (i = 0; i < m; i++)
   {
      j1 = A_i[i];
      j2 = A_i[i + 1];
      for (j = j1; j < j2; j++)
      {
         temp_matrix[i + A_j[j] * m] = 1;
      }
   }

   for (i = 0; i < m; i++)
   {
      for (j = 0; j < n; j++)
      {
         printf("%d ", temp_matrix[i + j * m]);
      }
      printf("\n");
   }

   AATGS_FREE(temp_matrix);
}

void AatgsTestPrintCSRMatrixToFile(FILE *file, int *A_i, int *A_j, AATGS_DOUBLE *A_a, int m, int n)
{
   AATGS_DOUBLE *temp_matrix = NULL;
   AATGS_CALLOC(temp_matrix, (size_t)m * n, AATGS_DOUBLE);

   int i, j1, j2, j;
   for (i = 0; i < m; i++)
   {
      j1 = A_i[i];
      j2 = A_i[i + 1];
      for (j = j1; j < j2; j++)
      {
         temp_matrix[i + A_j[j] * m] = A_a[j];
      }
   }

   for (i = 0; i < m; i++)
   {
      for (j = 0; j < n; j++)
      {
         fprintf(file, "%16.6f", temp_matrix[i + j * m]);
      }
      fprintf(file, "\n");
   }

   AATGS_FREE(temp_matrix);
}

void AatgsTestPrintCSRMatrixVal(int *A_i, int *A_j, int m, int n, AATGS_DOUBLE *A_a)
{
   int field_width = 10; // Adjust this as needed
   int i, j, k = 0;
   for (i = 0; i < m; i++)
   {
      for (j = 0; j < n; j++)
      {
         if (j == A_j[k] && k < A_i[i + 1])
         {
            printf("%*f ", field_width, A_a[k]);
            k++;
         }
         else
         {
            printf("%*d ", field_width, 0);
         }
      }
      printf("\n");
   }
}