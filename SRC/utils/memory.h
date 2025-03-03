#ifndef AATGS_MEMORY_H
#define AATGS_MEMORY_H

/**
 * @file memory.h
 * @brief Memory management
 */

#include "utils.h"

#define AATGS_MALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) AatgsMalloc( (size_t)(length)*sizeof(__VA_ARGS__));\
}

#define AATGS_CALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) AatgsCalloc( (size_t)(length)*sizeof(__VA_ARGS__), 1);\
}

#define AATGS_REALLOC(ptr, length, ...) {\
   (ptr) = (__VA_ARGS__*) AatgsRealloc( (void*)(ptr), (size_t)(length)*sizeof(__VA_ARGS__));\
}

#define AATGS_MEMCPY(ptr_to, ptr_from, length, ...) {\
   AatgsMemcpy( (void*)(ptr_to), (void*)(ptr_from), (size_t)(length)*sizeof(__VA_ARGS__));\
}

#define AATGS_FREE( ptr) {\
   if(ptr){AatgsFreeHost( (void*)(ptr));}\
   (ptr) = NULL;\
}

/**
 * @brief Allocate memory on host
 * @details Allocate memory on host
 * @param[in] size Size of memory to be allocated
 * @return Pointer to allocated memory
 */
static inline void* AatgsMalloc(size_t size)
{
   void *ptr = NULL;
   ptr = malloc(size);
   return ptr;
}

/**
 * @brief Allocate memory on host and initialize to zero
 * @details Allocate memory on host and initialize to zero
 * @param[in] length Length of memory to be allocated
 * @param[in] unitsize Size of each unit of memory
 * @return Pointer to allocated memory
 */
static inline void* AatgsCalloc(size_t length, int unitsize)
{
   void *ptr = NULL;
   ptr = calloc(length, unitsize);
   return ptr;
}

/**
 * @brief Reallocate memory on host
 * @details Reallocate memory on host
 * @param[in,out] ptr Pointer to memory to be reallocated
 * @param[in] size Size of memory to be allocated
 * @return Pointer to allocated memory
 */
static inline void* AatgsRealloc(void *ptr, size_t size)
{
   return ptr ? realloc( ptr, size ) : malloc( size );
}

/**
 * @brief Copy memory on host
 * @details Copy memory on host
 * @param[in,out] ptr_to Pointer to memory to be copied to
 * @param[in] ptr_from Pointer to memory to be copied from
 * @param[in] size Size of memory to be copied
 */
static inline void AatgsMemcpy(void *ptr_to, void *ptr_from, size_t size)
{
#ifdef AATGS_USING_OPENMP
#ifndef AATGS_OPENMP_NO_MEMCPY
   // use openmp to copy if possible, might not gain on all systems
   if(!omp_in_parallel())
   {
      size_t i;
      #pragma omp parallel for AATGS_DEFAULT_OPENMP_SCHEDULE
      for(i = 0; i < size; i++)
      {
         ((char*)ptr_to)[i] = ((char*)ptr_from)[i];
      }
      return;
   }
#endif
#endif
   memcpy( ptr_to, ptr_from, size);
}

/**
 * @brief Free memory on host
 * @details Free memory on host
 * @param[in,out] ptr Pointer to memory to be freed
 */
static inline void AatgsFreeHost(void *ptr)
{
   free(ptr);
}

#endif