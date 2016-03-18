#ifndef CNN_ALIGNED_MEM_POOL_H
#define CNN_ALIGNED_MEM_POOL_H

#include <cstdlib>
#include <cstring>
#include <iostream>

#ifdef WIN32
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif
#include "cnn/except.h"
#if HAVE_CUDA
#include "cnn/cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace cnn {

inline void* cnn_mm_malloc(size_t n, size_t align, bool on_cpu_only = false) {
  void* ptr = nullptr;
  if (!on_cpu_only)
  {
#if HAVE_CUDA
      CUDA_CHECK(cudaMalloc(&ptr, n));
#else
      ptr = _mm_malloc(n, align);
#endif
  }
  else
  {
      ptr = _mm_malloc(n, align);
  }
  if (!ptr) {
    std::cerr << "Memory allocation failed n=" << n << " align=" << align << std::endl;
    throw cnn::out_of_memory("Memory allocation failed in cnn_mm_malloc()");
  }
  return ptr;
}

inline void cnn_mm_free(void* mem, bool on_cpu_only = false) {
    if (on_cpu_only)
        _mm_free(mem);
    else{
#if HAVE_CUDA
        CUDA_CHECK(cudaFree(mem));
#else
        _mm_free(mem);
#endif
    }
}

inline void* cnn_mm_malloc_host(size_t n, size_t align) {
    void* ptr = nullptr;
#if HAVE_CUDA
    CUDA_CHECK(cudaMallocHost(&ptr, n));
#else
    ptr = _mm_malloc(n, align);
#endif
    if (!ptr) {
        std::cerr << "Memory allocation failed n=" << n << " align=" << align << std::endl;
        throw cnn::out_of_memory("Memory allocation failed in cnn_mm_malloc()");
    }
    return ptr;
}

inline void cnn_mm_free_host(void* mem) {
#if HAVE_CUDA
    CUDA_CHECK(cudaFreeHost(mem));
#else
    _mm_free(mem);
#endif

    //#else
    //  return std::free(n, align);
    //#endif
}

// this is used to manage CPU memory for function values and gradients
template <unsigned AlignedBits>
class AlignedMemoryPool {
 private:
  bool mb_allocate_on_cpu_only; 
 public:
  explicit AlignedMemoryPool(unsigned long cap, bool b_allocate_on_cpu_only = false) {
      mem = nullptr;
      mb_allocate_on_cpu_only = b_allocate_on_cpu_only;
      sys_alloc(cap);
      zero_all();
  }
  ~AlignedMemoryPool()
  {
      if (mem)
          cnn_mm_free(mem, mb_allocate_on_cpu_only); 
  }

  // returns nullptr if OOM
  void* allocate(unsigned long n) {
    auto rounded_n = round_up_align(n);
    if (rounded_n + used > capacity)
    {
        std::runtime_error("cannot allocate enough space");
        return nullptr;
    }
    void * res = static_cast<char*>(mem)+used;
    used += rounded_n;
    return res;
  }
  void free() {
    //std::cerr << "freeing " << used << " bytes\n";
    used = 0;
  }
  void free_and_grow_capacity(unsigned long new_cap = 0) {
    cnn_mm_free(mem, mb_allocate_on_cpu_only);
    if (new_cap)
      sys_alloc(new_cap);
    else
      sys_alloc(capacity * 1.5);
    zero_all();
  }
  // zeros out the amount of allocations
  void zero_allocated_memory() {
    if (used == 0) return;
    if (mb_allocate_on_cpu_only)
        std::memset(mem, 0, used);
    else{
#if HAVE_CUDA
        CUDA_CHECK(cudaMemsetAsync(mem, 0, used));
#else
        std::memset(mem, 0, used);
#endif
    }
  }
 private:
  void sys_alloc(unsigned long cap) {
    capacity = round_up_align(cap);
    mem = cnn_mm_malloc(capacity, 1 << AlignedBits, mb_allocate_on_cpu_only);
    used = 0;
  }
  void zero_all() {
      if (mb_allocate_on_cpu_only)
          std::memset(mem, 0, capacity);
      else{
#if HAVE_CUDA
          CUDA_CHECK(cudaMemsetAsync(mem, 0, capacity));
#else
          std::memset(mem, 0, capacity);
#endif
      }
  }
  inline static size_t round_up_align(unsigned long n) {
    if (AlignedBits < 2) return n;
    auto c = (n & ((1 << (AlignedBits)) - 1)) > 0 ? 1 : 0;
    return ((n >> (AlignedBits)) + c) << (AlignedBits);
  }
  unsigned long capacity;
  unsigned long used;
  void* mem;
};

} // namespace cnn

#endif
