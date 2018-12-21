/***************************************************************************
*  FILE NAME	: convolution.cu
*
*  DESCRIPTION  : CUDA Program to perform image convolution.
* 
*  PLATFORM		: Linux
*
*  DATE	               	NAME	        	  	REASON
*  5th Dec,2018         Shashi Shivaraju        CPSC_6780_Final_Project
*                       [C88650674]
****************************************************************************/
/*Header file inclusions*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

/*Macro declarations*/
#define THREADS_PER_BLOCK 512           /*number of threads per block*/
#define MAX_KERNEL_SIZE		20
#undef DEBUG_MODE


/*Constant memory to store the kernel since all threads will be accessing it*/
__constant__ float CKernel[MAX_KERNEL_SIZE*MAX_KERNEL_SIZE];


/*Function prototypes*/

/*Function to read a PPM image file*/
unsigned char ** ReadImage(char* inputfile,int *prows,int *pcols);
/*Function to read a filter file*/
float ** ReadFilter(char *filterfile,int *filtersize,float *scalefactor);
/*Function to perform convolution at CPU*/
int CPU_ConvolveImage(unsigned char **image,int cols,int rows,
					  float **filter,int kernel_size,float kernel_scalefactor,
					  unsigned char ** convoluted_image);				  
/*Function to write a PPM image file*/
int WriteImage(char *filename,unsigned char **image,int rows,int cols);

/*Wrapper function to perform convolution at GPU*/
int GPU_ConvolveWrapper(unsigned char **image,int cols,int rows,
					  float **filter,int kernel_size,float kernel_scalefactor,
					  unsigned char ** convoluted_image);
					  
/*kernel to perform convolution at GPU*/					  
__global__ void GPU_ConvolveKernel(unsigned char *image,int cols,int rows,
									int kernel_size,float kernel_scalefactor,
									float *inter_image);

/*main function of the program*/
int main(int argc,char* argv[])
{
	
	unsigned char **image = NULL;	/*pointer to store input image data*/
	unsigned char **output = NULL; 	/*pointer to store output image data*/
	float **filter = NULL; 			/*pointer to store the filter data*/
	int ret = 0;
	int row=0;                      /*variable to store height of input image*/
	int col=0;                      /*variable to store width of input image*/
	int filtersize = 0;				/*filter is always square*/
	float scalefactor = 0;			/*filter scale factor*/
	float CPU_milliseconds = 0,GPU_milliseconds = 0;			/*time profilling variables*/
	float speedup = 0;				/*GPU speedup*/
	cudaEvent_t start, stop;			/*using cuda events to time the kernel functions*/
	
	/*check for valid cmd line args*/
	if(argc != 3)
	{
		printf("[Usage]:./exe input.ppm filter.filt");
		return -1;
	}
	
	/*read the input image*/
	image = ReadImage(argv[1],&row,&col);
	if(!image)
	{
		printf("ReadImage failed\n");
		ret = -1;
		goto CLEANUP;
	}
	
	/*read the filter*/
	filter = ReadFilter(argv[2],&filtersize,&scalefactor);
	if(!filter || 0 == filtersize)
	{
		printf("ReadFilter failed\n");
		ret = -1;
		goto CLEANUP;
	}
	
	/*check if filtersize can be supoorted*/
	if(filtersize > MAX_KERNEL_SIZE)
	{
		printf("Max Kernel size = %d;Cannot support the Kernel size = %d;\
		Reduce the kernel size\n",MAX_KERNEL_SIZE,filtersize);
		ret = -1;
		goto CLEANUP;
	}
	
	/*create cuda events*/
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	/*allocate memory for the output image*/
	output = (unsigned char **)calloc(row,sizeof(unsigned char *));
	if(!output)			/*Error handling*/
	{
		printf("calloc failed\n");/*Memory allocation failed*/
		ret = -1;
		goto CLEANUP;
	}
	output[0] = (unsigned char *)calloc(row*col,sizeof(unsigned char));/*Allocate memory to store input image data*/
	if(!output[0])			/*Error handling*/
	{
		printf("calloc failed\n");/*Memory allocation failed*/
		ret = -1;
		goto CLEANUP;
	} 
	
	/*update the pointers to point to each row of the image in continous memory allocation */
	for (int i = 1; i < row; i++)
		output[i] = output[i - 1] + col;
	
	
	/*Begin time profilling for CPU Convolution*/  
	cudaEventRecord(start);
	
	/*Perform Convolution at CPU*/
	ret = CPU_ConvolveImage(image,col,row,filter,filtersize,scalefactor,output);
	if(ret != 0)
	{
		goto CLEANUP;
	}
	
	/*End time profiling for CPU Convolution*/  
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&CPU_milliseconds, start, stop);
	
	printf("CPU took %f milliseconds and output stored as CPU_Output.ppm \n",CPU_milliseconds);
	
	/*write the image to file*/
	WriteImage((char *)"CPU_Output.ppm",output,row,col);
	
	/*clear the output image*/
	memset(output[0],0,row*col);
	
	/*Begin time profilling for CPU Convolution*/  
	cudaEventRecord(start);
	
	/*call wrapper function to perform convolution at GPU*/
	ret = GPU_ConvolveWrapper(image,col,row,filter,filtersize,scalefactor,output);
	if(ret != 0)
	{
		goto CLEANUP;
	}
	
	/*End time profiling for CPU Convolution*/  
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_milliseconds, start, stop);
	
	printf("GPU took %f milliseconds and output stored as GPU_Output.ppm \n",GPU_milliseconds);
	speedup = GPU_milliseconds/CPU_milliseconds;
	printf("SpeedUP = GPU time/CPU time = %f\n",speedup);
	/*write the image to file*/
	WriteImage((char *)"GPU_Output.ppm",output,row,col);
	
CLEANUP:

	/*free all the heap allocations*/
	if(image)
	{
		if(image[0])
			free(image[0]);
			
		free(image);
		image = NULL;
	}
	
	if(output)
	{
		if(output[0])
			free(output[0]);
			
		free(output);
		output = NULL;
	}
		
	if(filter)
	{
		if(filter[0])
			free(filter[0]);
				
		free(filter);
		filter = NULL;
	}
	
	return ret;
}


/*Function Definations*/

/*Function to read a PPM image file*/
unsigned char ** ReadImage(char* inputfile,int *prows,int *pcols)
{
	FILE* fp = NULL;                /*File pointer for file operations*/
	unsigned char **image = NULL;	/*pointer to store input image data*/
	unsigned char *tempImage = NULL; /*temporary pointer to store the pixel map*/
	int ret = 0;
	char magic[10];                 /*array to store header of the input image*/
	int row=0;                      /*variable to store height of input image*/
	int col=0;                      /*variable to store width of input image*/
	int max_pixel=0;                /*maximum pixel value of the image*/
	
	fp = fopen(inputfile,"r");        /*open input image file provided as cmd line arg*/
	if(!fp)                         /*error handling*/
	{
		printf("fopen failed for %s\n",inputfile);/*failure to open the input file*/
		return NULL;              /*return error code*/	
	}
	
	ret = fscanf(fp,"%s %d %d %d "
	,magic,&col,&row,&max_pixel);	/*read header information of the image*/
	if(	4 != ret || 
		255 != max_pixel ||
		0 != strcmp("P5",magic))/*error handling specific to 8bit greyscale PPM image*/
	{
		printf("Not a greyscale image of PPM format\n");/*Not 8bit greyscale PPM image */
		return NULL;              /*return error code*/	
	}
	
	//row = 100;
	//col = 100;
	
	tempImage = (unsigned char *)calloc(row*col,sizeof(unsigned char));/*Allocate memory to store input image data*/
	if(!tempImage)			/*Error handling*/
	{
		printf("calloc failed\n");/*Memory allocation failed*/
		return NULL;              /*return error code*/
	}
	
	ret = fread(tempImage,1,row*col,fp);/*read the image data form file and store in a buffer*/
	if(ret != row*col)              /*check for invalid fread*/
	{
		printf("fread failed to read %d data from file",row*col); /*fread operation failed*/
		return NULL;              /*return error code*/
	}

	if(fp)                          /*close the file handle of  input file*/
	{
		fclose(fp);
		fp = NULL;
	}
	
	image = (unsigned char **)calloc(row,sizeof(unsigned char *));
	if(!image)			/*Error handling*/
	{
		printf("calloc failed\n");/*Memory allocation failed*/
		return NULL;              /*return error code*/
	}
	
	image[0] = (unsigned char *)calloc(row*col,sizeof(unsigned char));/*Allocate memory to store input image data*/
	if(!image[0])			/*Error handling*/
	{
		printf("calloc failed\n");/*Memory allocation failed*/
		return NULL;              /*return error code*/
	} 
	
	/*update the pointers to point to each row of the image in continous memory allocation */
	for (int i = 1; i < row; i++)
		image[i] = image[i - 1] + col;
		
	/*copy the pixels directly*/
	memcpy(image[0],tempImage,row*col);

#ifdef DEBUG_MODE
	printf("Image dimension: Width = %d Height = %d\n",col,row);
#endif /*DEBUG_MODE*/
	
	*prows = row;
	*pcols = col;

	/*free the memory*/
	if(tempImage)
	 free(tempImage);
	tempImage = NULL;
		
	return image;
}


/*Function to read the filter weights from file*/
float ** ReadFilter(char *filterfile,int *filtersize,float *scalefactor)
{
	FILE* fp = NULL;                /*File pointer for file operations*/
	float **kernel_weights = NULL;
	float **temp_weights = NULL;
	int r=0,c=0;
	int kernel_size = 0;
	float kernel_scalefactor = 1;
	float positive_weights = 0;
	float negative_weights = 0;
	
	fp = fopen(filterfile,"r");        /*open filter file provided as cmd line arg*/
	if(!fp)                         /*error handling*/
	{
		printf("fopen failed for %s\n",filterfile);/*failure to open the input file*/
		return NULL;              /*return error code*/	
	}
	
	
	fscanf(fp,"%d",filtersize);
#ifdef DEBUG_MODE
	printf("Filter Size:%d\n",*filtersize);
#endif /*DEBUG_MODE*/	

	kernel_size = *filtersize;

	 /*store the kernel*/
	 kernel_weights = (float **)calloc(kernel_size,sizeof(float *));
	 temp_weights = (float **)calloc(kernel_size,sizeof(float *));
	 
	 kernel_weights[0] = (float *)calloc(kernel_size*kernel_size,sizeof(float)); 
	 temp_weights[0] = (float *)calloc(kernel_size*kernel_size,sizeof(float));
	  for (int i = 1; i < kernel_size; i++)
	  {
			kernel_weights[i] = kernel_weights[i - 1] + kernel_size;
			temp_weights[i]   = temp_weights[i-1] + kernel_size;
	  }
	
	 kernel_scalefactor = 0;

#ifdef  DEBUG_MODE
	   printf("Kernel from the file:\n");
#endif /*DEBUG_MODE*/
	  	
	 /*read the  weights of kernel and determine scale factor*/
	 for (r = 0; r < kernel_size; r++)
	 {
		for (c = 0; c < kernel_size; c++)
		{
			fscanf(fp,"%f",&kernel_weights[r][c]);
			
			if(0 < kernel_weights[r][c])
			{
				positive_weights = positive_weights + kernel_weights[r][c];
			}
			else
			{
				negative_weights = positive_weights + kernel_weights[r][c];
			}
#ifdef  DEBUG_MODE
			printf("%lf\t",kernel_weights[r][c]);
#endif /*DEBUG_MODE*/
		}
#ifdef  DEBUG_MODE
		printf("\n");
#endif /*DEBUG_MODE*/
	 }
	 
	 /*The scale factor to use is the maximum magnitude of either 
	  * (a) the sum of the positive weights or 
	  * (b) the sum of the negative weights.*/
	  kernel_scalefactor = (abs(positive_weights) > abs(negative_weights))?positive_weights:abs(negative_weights);
	  
	 
#ifdef  DEBUG_MODE
		printf("kernel scale factor = %f\n",kernel_scalefactor);
#endif /*DEBUG_MODE*/

		*scalefactor = kernel_scalefactor;
	 
	
	 /*flip the kernel horizontally*/
	 for (r = 0; r < kernel_size; r++)
	 {
		for (c = 0; c < kernel_size; c++)
		{
			temp_weights[r][c] = kernel_weights[r][kernel_size-1-c];

		}
	 }
	 
#ifdef  DEBUG_MODE
	   printf("Final Kernel after flipping:\n");
#endif /*DEBUG_MODE*/
	 
	 /*flip the kernel vertically*/
	 for (r = 0; r < kernel_size; r++)
	 {
		for (c = 0; c < kernel_size; c++)
		{
			kernel_weights[r][c] = temp_weights[kernel_size-1-r][c];
#ifdef  DEBUG_MODE
			printf("%lf\t",kernel_weights[r][c]);
#endif /*DEBUG_MODE*/
		}
#ifdef  DEBUG_MODE
				printf("\n");
#endif /*DEBUG_MODE*/
	 }

	/*deallocate the memory*/
	if(temp_weights)
	{
		free(temp_weights[0]);
		free(temp_weights);
	}
	
	/*close the file*/
	if(fp)                          /*close the file handle of  input file*/
	{
		fclose(fp);
		fp = NULL;
	}
	
	return kernel_weights;
}


/*Function to perform convolution at CPU sequentially*/
int CPU_ConvolveImage(unsigned char **image,int cols,int rows,
					  float **filter,int kernel_size,float kernel_scalefactor,
					  unsigned char ** convoluted_image)
{
	unsigned char ** input = image;
	float **inter_image = NULL;
	float pixel = 0;
	float sum = 0;
	float min = 0,max = 0;
	float range = 0;
	int border = kernel_size/2;
	int R = 0,C = 0,r=0,c=0; 
	int count = 0;
	
	/*allocate memory to store intermediate float image*/
	inter_image = (float **)calloc(rows,sizeof(float *));
	inter_image[0] = (float *)calloc(rows*cols,sizeof(float));;
	/*update the pointers to point to each row of the image in continous memory allocation */
	for (int i = 1; i < rows; i++)
		inter_image[i] = inter_image[i - 1] + cols;
	
	/*perform convolution of the image with kernel by excluding the borders*/
	for(R = 0;R < rows;R++)
	{
		for(C=0;C < cols;C++)
		{
			sum = 0;
			for(r=-border;r<=border;r++)
			{
				for(c=-border;c<=border;c++)
				{
					/*check if pixel index is outside the image*/
					if(R+r < 0 || R+r >= rows || C+c < 0 || C+c >= cols )
					{
						continue;
					}
					sum = sum + input[R+r][C+c]*filter[r+border][c+border];
				}
			}
			
			inter_image[R][C] = sum/(float)kernel_scalefactor;
			
			if(!count|| inter_image[R][C] < min)
				min = inter_image[R][C];

			if(!count|| inter_image[R][C] > max)
				max = inter_image[R][C];
				
			count ++;
		}
	}
	
	/*Normalize the results*/
	range = max-min;
	
	for(R = 0;R < rows;R++)
	{
		for(C = 0;C < cols;C++)
		{
			pixel = inter_image[R][C];
			
			pixel = ((pixel-min)/range)*255;
			
			/*rounding the values to the next highest integer if value above 0.5  */
			if(pixel-(unsigned char)pixel > 0.5)
				convoluted_image[R][C] = (unsigned char)(pixel+1);
			else
				convoluted_image[R][C] = (unsigned char)pixel;
		}
	}
	
	return 0;	
}



/*Function to write a PPM image file*/
int WriteImage(char * filename,unsigned char **image,int row,int col)
{
	FILE* fp = NULL;                /*File pointer for file operations*/
	
	fp = fopen(filename,"w+");/*open output image file*/
	if(!fp)                     /*error handling*/
	{
		printf("fopen failed for %s\n",filename);/*failure to open the output file*/
		return -1;                  /*return error code*/
	}
	
	fprintf(fp,"P5 %d %d 255 ",col,row);/*Write the header as per PPM image specification to output image file*/
	fwrite(image[0],1,row*col,fp);/*write the output image data into file*/
	
	if(fp)                      /*Close the output file handle*/
	{
		fclose(fp);
		fp = NULL;
    }
    
    return 0;	
}


/*Wrapper function to perform convolution at GPU*/
int GPU_ConvolveWrapper(unsigned char **image,int col,int row,
					  float **filter,int kernel_size,float kernel_scalefactor,
					  unsigned char ** convoluted_image)
					  
{
	unsigned char *Dimage = NULL;	/*pointer to store input image data*/
	float *Dinter = NULL; 	/*pointer to store output image data*/
	float **inter_image = NULL;	
	float pixel = 0;
	float range = 0;
	float min = 0,max = 0;
	int R = 0,C = 0; 
	int ret = 0,i = 0;
	int nBlocks = 1;
	
	
	/*Allocate the GPU memory for the input image*/
	if(cudaSuccess != cudaMalloc((void **)&Dimage,sizeof(unsigned char)*row*col))
	{
		printf("CUDA memory allocation failed 1aa\n");
		ret = -1;
		goto CLEANUP; 
	}
		
	 /*Assign input image data to GPU memory for input image*/
	ret = cudaMemcpy(Dimage,image[0],sizeof(unsigned char)*row*col,cudaMemcpyHostToDevice);
	if(cudaSuccess != ret)
	{
		printf("cudaMemcpy failed %d\n",ret);
		ret = -1;
		goto CLEANUP;
	}
	
	/*Allocate the GPU memory for the output image*/
	if(cudaSuccess != cudaMalloc((void **)&Dinter,sizeof(float)*row*col))
	{
		printf("CUDA memory allocation failed 1aa\n");
		ret = -1;
		goto CLEANUP; 
	}
	
	ret = cudaMemcpyToSymbol(CKernel,filter[0],sizeof(float)*kernel_size*kernel_size);
	if(cudaSuccess != ret)
	{
		printf("cudaMemcpy failed %d\n",ret);
		ret = -1;
		goto CLEANUP;
	}
	
	/*limit number of blocks to 32*/
	nBlocks = (row*col + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	if(32 < nBlocks) 
		nBlocks = 32;
	
	/*Invoke kernel function to perform convolution*/
	GPU_ConvolveKernel<<<nBlocks,THREADS_PER_BLOCK>>>(Dimage,col,row,kernel_size,kernel_scalefactor,Dinter);
	
	 /*allocate CPU memory to store the intermediate results*/
	inter_image = (float **)calloc(row,sizeof(float *));
	inter_image[0] = (float *)calloc(row*col,sizeof(float));;
	/*update the pointers to point to each row of the image in continous memory allocation */
	for (int i = 1; i < row; i++)
		inter_image[i] = inter_image[i - 1] + col;
	
	 /*Assign GPU memory for inter image to inter image data at CPU*/
	ret = cudaMemcpy(inter_image[0],Dinter,sizeof(float)*row*col,cudaMemcpyDeviceToHost);
	if(cudaSuccess != ret)
	{
		printf("cudaMemcpy failed ssds %d\n",ret);
		ret = -1;
		goto CLEANUP;
	}
	
	for(i=0;i<row*col;i++)
	{
		if(!i|| inter_image[0][i] < min)
			min = inter_image[0][i];

		if(!i|| inter_image[0][i] > max)
			max = inter_image[0][i];
	}
	
	/*Normalize the results*/
	range = max-min;
	
	for(R = 0;R < row;R++)
	{
		for(C = 0;C < col;C++)
		{
			pixel = inter_image[R][C];
			
			if(range)
				pixel = ((pixel-min)/range)*255;
			
			/*rounding the values to the next highest integer if value above 0.5  */
			if(pixel-(unsigned char)pixel > 0.5)
				convoluted_image[R][C] = (unsigned char)(pixel+1);
			else
				convoluted_image[R][C] = (unsigned char)pixel;
		}
	}
	
	
CLEANUP:

	if(inter_image)
	{
		if(inter_image[0])
			free(inter_image[0]);
			
		free(inter_image);
		inter_image = NULL;
	}
	
	if(Dinter)
	{		
		cudaFree(Dinter);
		Dinter = NULL;
	}
	
	if(Dimage)
	{		
		cudaFree(Dimage);
		Dimage = NULL;
	}
	
	return ret;
}

/*Kernel function to perform convolution at the GPU*/
__global__ void GPU_ConvolveKernel(unsigned char *image,int cols,int rows,
									int kernel_size,float kernel_scalefactor,
									float *convoluted_image)
{	
	float sum = 0;
	int border = kernel_size/2;
	int r = 0,c = 0,R = 0,C = 0;
	
	/*Calculate index of the pixel*/
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (index < rows*cols)
	{
		/*convert index into position in raster*/
		R = index/cols;
		C = index -(R*cols);
		
		sum = 0;
		if(R >= 0 && R < rows && C >= 0 && C < cols)
		{
			for(r=-border;r<=border;r++)
			{
				for(c=-border;c<=border;c++)
				{
					/*check if pixel index is outside the image*/
					if(R+r < 0 || R+r >= rows || C+c < 0 || C+c >= cols )
					{
						continue;
					}
					sum = sum + image[(R+r)*cols+(C+c)]*CKernel[(r+border)*3+(c+border)];
				}
			}		
			convoluted_image[index] = sum/(float)kernel_scalefactor;
		}
		/*utilize the same thread to process another pixel if possible*/
		index += blockDim.x * gridDim.x;
	}
}


