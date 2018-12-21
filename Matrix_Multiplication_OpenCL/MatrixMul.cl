/***************************************************************************
*  FILE NAME	: MatrixMul.cl
*
*  DESCRIPTION  : OpenCL kernel to perform multiplication of two square 
* 				  matrix.
* 
*  PLATFORM		: Linux
*
*  DATE	               	NAME	        	  	REASON
*  9th Nov,2018         Shashi Shivaraju        CPSC_6780_Assignment_03
*                       [C88650674]
*
****************************************************************************/
__kernel void MatrixMul(__global float* A,__global float* B, __global float* AB,int order)
{
  /*unique global work-item ID value for dimension*/
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   float value = 0;
   for (int k = 0; k < order; ++k)
   {
      float elementA = A[ty * order + k];
      float elementB = B[k * order + tx];
      value += elementA * elementB;
   }
   
   AB[ty * order + tx] = value;
}
