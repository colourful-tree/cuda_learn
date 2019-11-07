#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

int N = 64; 

__global__ void VecAddOne(int* A) {
    int i = threadIdx.x;
    A[i] = A[i] + 1;
}


bool test_cudaHostAlloc() {

  bool res = true;
  int *dptr1, *hptr1;

  auto start = std::chrono::system_clock::now();

  // First way: using cudaHostAlloc()
  // Alloc host page-locked memory
  (cudaHostAlloc(&hptr1, sizeof(int) * N, cudaHostAllocMapped));

  // Get corresponding device pointer
  (cudaHostGetDevicePointer(&dptr1, hptr1, 0));

  // Initialize this memory
  for(int i = 0; i < N; i++)
    hptr1[i] = i;

  // Test kernel
  VecAddOne<<<1, N>>>(dptr1);
  cudaDeviceSynchronize();

  // Check result
  for(int i = 0; i < N; i++){
    //printf("s[%d]: %d\n", i, hptr1[i]);
    res = (hptr1[i] == i + 1) ? res : false;
  }
  // Free memory
  (cudaFreeHost(hptr1));
  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "cudaHostAlloc time:" << duration.count() << std::endl;

  return res;
}

bool test_cudaHostRegister() {

  bool res = true;
  int *dptr1, *hptr1;

  // Second way: using cudaHostRegister()
  // Alloc host memory
  hptr1 = new int[N];
  assert(hptr1);

  // Initialize this memory
  for(int i = 0; i < N; i++)
    hptr1[i] = i;

  auto start = std::chrono::system_clock::now();
  // Page-lock host memory
  cudaHostRegister(hptr1, N, cudaHostRegisterMapped);

  // Get corresponding device pointer
  (cudaHostGetDevicePointer(&dptr1, hptr1, 0));

  // Test kernel
  if(cudaDevAttrCanUseHostPointerForRegisteredMem != 0){
    printf("Can directly use host pointer to substitute device pointer on this machine.\n");
    VecAddOne<<<1, N>>>(hptr1);
  }
  else{
    printf("This machine does not support substituting host pointer for device pointer.\n");
    VecAddOne<<<1, N>>>(dptr1);
  }
  cudaDeviceSynchronize();
  // Free memory
  cudaHostUnregister(hptr1);
  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "cudaHostRegister time:" << duration.count() << std::endl;

  // Check result
  for(int i = 0; i < N; i++){
    //printf("s[%d]: %d\n", i, hptr1[i]);
    res = (hptr1[i] == i + 1) ? res : false;
  }
  delete hptr1;
  return res;
}

bool test_cudaMemcpy() {

  bool res = true;
  int *dptr1, *hptr1;

  // Alloc host memory
  hptr1 = new int[N];
  assert(hptr1);

  // Initialize this memory
  for(int i = 0; i < N; i++)
    hptr1[i] = i;

  auto start = std::chrono::system_clock::now();
  cudaMalloc(&dptr1, sizeof(int) * N);
  cudaMemcpy(dptr1, hptr1, sizeof(int) * N, cudaMemcpyHostToDevice);

  VecAddOne<<<1, N>>>(dptr1);
  cudaDeviceSynchronize();
  cudaMemcpy(hptr1, dptr1, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaFree(dptr1);

  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "cudaMemcpy time:" << duration.count() << std::endl;

  // Check result
  for(int i = 0; i < N; i++){
    // printf("s[%d]: %d\n", i, hptr1[i]);
    res = (hptr1[i] == i + 1) ? res : false;
  }

  // Free memory
  delete hptr1;
  return res;
}

bool test_cudaMemcpyAsync() {

  bool res = true;
  int *dptr1, *hptr1;

  // Alloc host memory
  hptr1 = new int[N];
  assert(hptr1);

  // Initialize this memory
  for(int i = 0; i < N; i++)
    hptr1[i] = i;
  
  auto start = std::chrono::system_clock::now();

  cudaMalloc(&dptr1, sizeof(int) * N);
  cudaMemcpyAsync(dptr1, hptr1, sizeof(int) * N, cudaMemcpyHostToDevice);

  VecAddOne<<<1, N>>>(dptr1);
  cudaDeviceSynchronize();
  cudaMemcpyAsync(hptr1, dptr1, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaFree(dptr1);

  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "cudaMemcpyAsync time:" << duration.count() << std::endl;

  // Check result
  for(int i = 0; i < N; i++){
    //printf("s[%d]: %d\n", i, hptr1[i]);
    res = (hptr1[i] == i + 1) ? res : false;
  }

  // Free memory
  delete hptr1;
  return res;
}

int main() {
  // Enable memory mapping
  // cudaSetDeviceFlags(cudaDeviceMapHost);
  // cout << test_cudaHostAlloc() << endl; //First will take long time, it's little confused.

  cout << test_cudaHostAlloc() << endl;
  cout << test_cudaHostRegister() << endl;
  cout << test_cudaMemcpy() << endl;
  cout << test_cudaMemcpyAsync() << endl;

  return 0;
}
