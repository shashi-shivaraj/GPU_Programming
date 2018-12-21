/***************************************************************************
*  FILE NAME	: big_dot_2.cu
*
*  DESCRIPTION  : CUDA Program to compute dot product of two vectors using 
* 				  parallel reduction and atomic operations.
* 
*  PLATFORM		: Linux
*
*  DATE	               	NAME	        	  	REASON
*  7th Oct,2018         Shashi Shivaraju        CPSC_66780_Assignment_02
*                       [C88650674]
****************************************************************************/
/*Header file inclusions*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

/*Macro declarations*/
#define THREADS_PER_BLOCK 512           /*number of threads per block*/

/*Function prototypes*/

/*Function to assign data to the vectors*/
void AssignVectorData(float* VecA,float* VecB,int N);


/*Function to calculate dot product of two floating point vectors*/
float CPU_big_dot(float *A,float *B,int N);

/*Function to invoke GPU_big_dot_Kernel_1*/
int  GPU_Kernel_1(float *Vec_A,float *Vec_B,int TOTAL_ELEMENTS,float *GPU_dotproduct);
/*Function to invoke GPU_big_dot_Kernel_2*/
int  GPU_Kernel_2(float *Vec_A,float *Vec_B,int TOTAL_ELEMENTS,float *GPU_dotproduct);


/*GPU Kernel functions*/
/*Kernel function to calculate partial sum on each thread block*/
__global__ void GPU_big_dot_Kernel_1(float *A,float *B,float *blocksum,int N);
/*Kernel function to calculate dot product at the GPU*/
__global__ void GPU_big_dot_Atomic_Kernel_2(float *A,float *B,float *Ddotproduct,int N);



/*main function of the program*/
int main()
{
  /*Variable Declarations*/
  float *Vec_A = NULL,*Vec_B = NULL;	/*pointers to store cpu memory*/
  float CPU_dotproduct = 0;				/*variable to store CPU computation result*/
  float GPU_dotproduct = 0;				/*variable to store CPU computation result*/
  int ret = 0;
  int TOTAL_ELEMENTS =  1 << 24;  		/*number of elements in the vector*/
  float milliseconds = 0;				/*time profilling variables*/
  cudaEvent_t start, stop;				/*using cuda events to time the kernel functions*/
  
  /*create cuda events*/
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
   /*Allocate CPU memory to store vector Vec_A*/
  Vec_A = (float*)calloc(TOTAL_ELEMENTS,sizeof(float));
  if(!Vec_A)
  {
	  printf("memory allocation failed\n");
	  ret = -1;
	  goto CLEANUP;
  }
  
  /*Allocate CPU memory to store vector Vec_B*/
  Vec_B = (float*)calloc(TOTAL_ELEMENTS,sizeof(float));
  if(!Vec_B)
  {
	  printf("memory allocation failed\n");
	  ret = -1;
	  goto CLEANUP;
	  
  }
  
  /*Assign data to CPU vectors*/
  AssignVectorData(Vec_A,Vec_B,TOTAL_ELEMENTS);
  
  /*Begin time profilling for GPU Kernel Version 1*/  
  cudaEventRecord(start);
  
  ret = GPU_Kernel_1(Vec_A,Vec_B,TOTAL_ELEMENTS,&GPU_dotproduct);
  if(0 != ret)
  {
	  printf("GPU_big_dot_Kernel_1 failed\n");
	  goto CLEANUP;
  }
  
  /*End time profiling for CPU memory allocation*/  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  printf("GPU_Kernel_1 took %f milliseconds\n",milliseconds);
  printf("GPU Kernel Version 1 dot product = %f\n",GPU_dotproduct);
  
  /*Begin time profilling for GPU Kernel Version 2*/  
  cudaEventRecord(start);
  
   ret = GPU_Kernel_2(Vec_A,Vec_B,TOTAL_ELEMENTS,&GPU_dotproduct);
   if(0 != ret)
   {
	   printf("GPU_big_dot_Kernel_2 failed\n");
	   goto CLEANUP;
   }
  /*End time profiling for CPU memory allocation*/  
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  printf("GPU_Atomic_Kernel_2 took %f milliseconds\n",milliseconds);
  printf("GPU Atomic Kernel Version 2 dot product = %f\n",GPU_dotproduct);


  /*compute dot product on CPU*/
  CPU_dotproduct = CPU_big_dot(Vec_A,Vec_B,TOTAL_ELEMENTS);
  printf("[Verification]:CPU dot product = %f\n\
(Results of CPU and GPU might differ due to floating point precision)\n",CPU_dotproduct);
  
  CLEANUP:
   	/*Deallocate the memory*/
   	
   	/*Destroy cuda events*/
   	if(start)
		cudaEventDestroy(start);
	
	if(stop)
		cudaEventDestroy(stop);
  
	if(Vec_A)
	{
		free(Vec_A);
		Vec_A = NULL;
	}
		
	if(Vec_B)
	{
		free(Vec_B);
		Vec_A = NULL;
	}
 
	return ret;
}

/*Function to assign data to vectors*/
void AssignVectorData(float* VecA,float* VecB,int N)
{
	int count = 0;

	/* Use current time as seed for random generator */
        srand(time(0)); 
	for(count = 0;count<N;count++)
	{
		/*Assigning random floating point numbers as vector elements */
		VecA[count] = float(rand()%N)/N; 
		VecB[count] = float(rand()%N)/N; 
	}
}

/*Kernel function to calculate blockwise dot product of two floating point vectors*/
__global__ void GPU_big_dot_Kernel_1(float *A,float *B,float *blocksum,int N)
{
	/*Calculate index of the array*/
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	/*Shared memory to store the pair wise product*/
	/*Shared memory is shared only among the threads within the block*/
	__shared__ float pairwise_product[THREADS_PER_BLOCK];
	
	/*Each thread will calculate the pair wise products 
	 * from the global memory and its sum will be stored 
	 * in the shared memory within the block with its index */
	float thread_sum = 0;
	while (index < N)
	{
		thread_sum += A[index] * B[index];
		index += blockDim.x * gridDim.x;
	}
	pairwise_product[threadIdx.x] = thread_sum;
	
	/*The threads within the block get synchronized*/
	__syncthreads();

	 /*Parallel reduction implemented to calculate 
	  * the block sum of the dot product*/
	 /*Let us use multiple threads of each block to 
	  * calculate the single sum of pairwise products within the block */
	int i = blockDim.x/2;
	while (i != 0)
	{
		/*Initiallly half the number of threads will be used*/
		if (threadIdx.x < i)
          pairwise_product[threadIdx.x] += pairwise_product[threadIdx.x + i];
        
        /*The threads within the block get synchronized*/
         __syncthreads();
         
         /*reduce the threads to be used*/
			i = i/2;
	}
	
	/*The sum of the pairwise products within the block 
	 * will be stored at index zero of the shared memory*/
	 
	/* Let the first thread of the block access this 
	 * block sum and store it in the global memory with
	 *  the index as block id*/
	if(0 == threadIdx.x)
	{
		blocksum[blockIdx.x] = pairwise_product[0];
	} 
}


/*Kernel function to calculate dot product of two floating point vectors*/
__global__ void GPU_big_dot_Atomic_Kernel_2(float *A,float *B,float *Ddotproduct,int N)
{
	/*Calculate index of the array*/
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	/*Shared memory to store the pair wise product*/
	/*Shared memory is shared only among the threads within the block*/
	__shared__ float pairwise_product[THREADS_PER_BLOCK];
	
	/*Each thread will calculate the pair wise products 
	 * from the global memory and its sum will be stored 
	 * in the shared memory within the block with its index */
	float thread_sum = 0;
	while (index < N)
	{
		thread_sum += A[index] * B[index];
		index += blockDim.x * gridDim.x;
	}
	pairwise_product[threadIdx.x] = thread_sum;
	
	/*The threads within the block get synchronized*/
	__syncthreads();

	 /*Parallel reduction implemented to calculate 
	  * the block sum of the dot product*/
	 /*Let us use multiple threads of each block to 
	  * calculate the single sum of pairwise products within the block */
	int i = blockDim.x/2;
	while (i != 0)
	{
		/*Initiallly half the number of threads will be used*/
		if (threadIdx.x < i)
          pairwise_product[threadIdx.x] += pairwise_product[threadIdx.x + i];
        
        /*The threads within the block get synchronized*/
         __syncthreads();
         
         /*reduce the threads to be used*/
			i = i/2;
	}
	
	/*The sum of the pairwise products within the block 
	 * will be stored at index zero of the shared memory*/
	 
	/* Let the first thread of the block access this 
	 * block sum and store it in the global memory 
	 * using atomic operation*/
	if(0 == threadIdx.x)
	{
		atomicAdd(Ddotproduct,pairwise_product[0]);
	}
	
}

/*Wrapper function to calculate dot product using Kernel 1 at GPU*/
int  GPU_Kernel_1(float *Vec_A,float *Vec_B,int TOTAL_ELEMENTS,float *dotproduct)
{
  
  float *DVec_A = NULL,*DVec_B = NULL; 	/*pointers to store GPU memory*/
  float *Ddotproduct = NULL,*Blockdotproduct = NULL;    /*pointer to store the GPU dot product of each block*/
  float GPU_dotproduct = 0;                             /*variable to store GPU computation result*/
  int   nBlocks = 0;					/*variable to store number of blocks*/
  int ret = 0,count = 0;
  
  /*Allocate the GPU memory to store vector DVec_A*/
  if(cudaSuccess != cudaMalloc(&DVec_A,sizeof(float)*TOTAL_ELEMENTS))
  {
  	printf("CUDA memory allocation failed\n");
	ret = -1;
	goto CLEANUP; 
  }

  /*Allocate the GPU memory to store vector DVec_B*/
  if(cudaSuccess != cudaMalloc(&DVec_B,sizeof(float)*TOTAL_ELEMENTS))
  {
	printf("CUDA memory allocation failed\n");
	ret = -1;
	goto CLEANUP; 
  }
  
  /*Assign data to GPU vectors*/
  ret = cudaMemcpy(DVec_A,Vec_A,sizeof(float)*TOTAL_ELEMENTS,cudaMemcpyHostToDevice);
  if(cudaSuccess != ret)
  {
 	printf("cudaMemcpy failed %d\n",ret);
	ret = -1;
	goto CLEANUP;
  }

  ret = cudaMemcpy(DVec_B,Vec_B,sizeof(float)*TOTAL_ELEMENTS,cudaMemcpyHostToDevice);
  if(cudaSuccess != ret)
  {
 	printf("cudaMemcpy failed %d\n",ret);
	ret = -1;
	goto CLEANUP;
  }
  
  /*restricting the number of blocks so that the threads in the block
   *can be utilized to work on multiple data in the global memory*/
   nBlocks = (TOTAL_ELEMENTS + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
   if(32 < nBlocks) 
	nBlocks = 32;

   GPU_dotproduct = 0;
   
    /*allocate memory to store dotproduct of each block*/
   if(cudaSuccess != cudaMalloc(&Ddotproduct,nBlocks*sizeof(float)))
   {
  	printf("CUDA memory allocation failed\n");
	ret = -1;
	goto CLEANUP; 
   }
	
   /*memset the GPU memory allocated */
   ret = cudaMemset(Ddotproduct,0,nBlocks*sizeof(float));
   if(cudaSuccess != ret)
   {
	printf("CUDA memset failed\n");
	ret = -1;
	goto CLEANUP; 
   }

   /*Allocate CPU memory to copy dotproduct of each block*/
   Blockdotproduct = (float*)calloc(nBlocks,sizeof(float));
   if(!Blockdotproduct)
   {
	  printf("memory allocation failed\n");
	  ret = -1;
	  goto CLEANUP;	  
   }
   
   /*Calculate the dot product of two vectors at GPU using Kernel 1*/
   GPU_big_dot_Kernel_1<<<nBlocks,THREADS_PER_BLOCK>>>(DVec_A,DVec_B,Ddotproduct,TOTAL_ELEMENTS);

   /*copy data from GPU to CPU*/   
   ret = cudaMemcpy(Blockdotproduct,Ddotproduct,nBlocks*sizeof(float),cudaMemcpyDeviceToHost);
   if(cudaSuccess != ret)
   {
 	printf("cudaMemcpy failed\n");
	ret = -1;
	goto CLEANUP;
   }
   
   /*calculate the dot product at CPU*/
   for(count = 0;count < nBlocks;count++)
   {
      GPU_dotproduct = GPU_dotproduct + Blockdotproduct[count];
   }
   
   *dotproduct = GPU_dotproduct;
   
CLEANUP:   
   
   if(Blockdotproduct)
	{
		free(Blockdotproduct);
		Blockdotproduct = NULL;
	}

	if(DVec_A)
	{
		cudaFree(DVec_A);
		DVec_A = NULL;
	}
	 
	if(DVec_B)
	{
		cudaFree(DVec_B);
		DVec_B = NULL;
	}

    if(Ddotproduct)
	{
		cudaFree(Ddotproduct);
		Ddotproduct = NULL;
	}
	
	return ret;
}

/*Wrapper function to invoke GPU_big_dot_Kernel_2*/
int  GPU_Kernel_2(float *Vec_A,float *Vec_B,int TOTAL_ELEMENTS,float *GPU_dotproduct)
{
  float *DVec_A = NULL,*DVec_B = NULL; 	/*pointers to store GPU memory*/
  float *Ddotproduct = NULL,*dotproduct = NULL;    /*pointer to store the GPU dot product*/
  int   nBlocks = 0;					/*variable to store number of blocks*/
  int ret = 0;
  
  /*Allocate the GPU memory to store vector DVec_A*/
  if(cudaSuccess != cudaMalloc(&DVec_A,sizeof(float)*TOTAL_ELEMENTS))
  {
  	printf("CUDA memory allocation failed\n");
	ret = -1;
	goto CLEANUP; 
  }

  /*Allocate the GPU memory to store vector DVec_B*/
  if(cudaSuccess != cudaMalloc(&DVec_B,sizeof(float)*TOTAL_ELEMENTS))
  {
	printf("CUDA memory allocation failed\n");
	ret = -1;
	goto CLEANUP; 
  }
  
  /*Assign data to GPU vectors*/
  ret = cudaMemcpy(DVec_A,Vec_A,sizeof(float)*TOTAL_ELEMENTS,cudaMemcpyHostToDevice);
  if(cudaSuccess != ret)
  {
 	printf("cudaMemcpy failed %d\n",ret);
	ret = -1;
	goto CLEANUP;
  }

  ret = cudaMemcpy(DVec_B,Vec_B,sizeof(float)*TOTAL_ELEMENTS,cudaMemcpyHostToDevice);
  if(cudaSuccess != ret)
  {
 	printf("cudaMemcpy failed %d\n",ret);
	ret = -1;
	goto CLEANUP;
  }
  
   /*restricting the number of blocks so that the threads
   *can be utilized to work on multiple data in the global memory*/
   nBlocks = (TOTAL_ELEMENTS + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
   if(32 < nBlocks) 
	nBlocks = 32;
   
    /*allocate memory to store dotproduct at GPU*/
   if(cudaSuccess != cudaMalloc(&Ddotproduct,1*sizeof(float)))
   {
  	printf("CUDA memory allocation failed\n");
	ret = -1;
	goto CLEANUP; 
   }
	
   /*memset the GPU memory allocated */
   ret = cudaMemset(Ddotproduct,0,1*sizeof(float));
   if(cudaSuccess != ret)
   {
	printf("CUDA memset failed\n");
	ret = -1;
	goto CLEANUP; 
   }

   /*allocate memory to store dotproduct at CPU*/
   dotproduct = (float*)calloc(1,sizeof(float));
   if(!dotproduct)
   {
	  printf("memory allocation failed\n");
	  ret = -1;
	  goto CLEANUP;	  
   }
   
   /*Calculate the dot product of two vectors at GPU using Kernel 2*/
   GPU_big_dot_Atomic_Kernel_2<<<nBlocks,THREADS_PER_BLOCK>>>(DVec_A,DVec_B,Ddotproduct,TOTAL_ELEMENTS);

   /*copy the dot product from GPU to CPU*/   
   ret = cudaMemcpy(dotproduct,Ddotproduct,1*sizeof(float),cudaMemcpyDeviceToHost);
   if(cudaSuccess != ret)
   {
 	printf("cudaMemcpy failed\n");
	ret = -1;
	goto CLEANUP;
   }
   
   *GPU_dotproduct = *dotproduct;
   
CLEANUP:   
   
   if(dotproduct)
	{
		free(dotproduct);
		dotproduct = NULL;
	}

	if(DVec_A)
	{
		cudaFree(DVec_A);
		DVec_A = NULL;
	}
	 
	if(DVec_B)
	{
		cudaFree(DVec_B);
		DVec_B = NULL;
	}

    if(Ddotproduct)
	{
		cudaFree(Ddotproduct);
		Ddotproduct = NULL;
	}
	
	return ret;
}

/*Function to calculate dot product of two floating point vectors*/
float CPU_big_dot(float *A,float *B,int N)
{
	int count = 0;
	float dot_product = 0;
	
	for(count = 0;count<N;count++)
	{
		dot_product = dot_product + A[count] * B[count];  
	}
	
	return dot_product;
}
