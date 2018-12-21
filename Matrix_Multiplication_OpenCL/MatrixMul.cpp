/***************************************************************************
*  FILE NAME	: MatrixMul.cpp
*
*  DESCRIPTION  : OpenCL Program to perform multiplication of two square 
* 				  matrix.
* 
*  PLATFORM		: Linux
*
*  DATE	               	NAME	        	  	REASON
*  9th Nov,2018         Shashi Shivaraju        CPSC_6780_Assignment_03
*                       [C88650674]
* 
* /usr/bin/g++ -O -I /usr/local/cuda/include MatrixMul.cpp 
* -L /usr/local/cuda/lib64/ 
* -lX11 -lGL -lGLU -lglut -lm -lXmu -lOpenCL -o MatrixMul
****************************************************************************/
/*Header file inclusions*/
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>

/*Macro declarations*/
#define N	40
#define BLOCK_SIZE	1
#undef DEBUG_MODE

/*Function prototypes*/
int CheckGPUSupport(cl_platform_id &platform_id,cl_device_id &device_id); 
/*Function to read the offline kernel file*/
char* loadProgSource(const char* filename, const char* preamble, size_t *sz);

/*main function of the program*/
int main()
{
	int ret = 0;
	cl_int err;
	cl_platform_id platform_id;
	cl_device_id device_id;
	cl_context_properties properties[3];
	cl_context context;
	cl_command_queue command_queue;
	cl_event prof_event;
	
	
	cl_float *inputMatrix1,*inputMatrix2,*result;
	cl_uint width = N;
	int x = 0 ,y = 0,data = 0;
	
	char *kernelSource;
	size_t kernelSize;
	cl_program program;
    cl_kernel kernel;
    
    cl_mem inputMat1,inputMat2,outputMat;
	
	/*Determine if the platform supports GPU and get platform ID and device ID*/
	ret = CheckGPUSupport(platform_id,device_id); 
	if(ret != 0)
	{
		printf("System doesnot support GPU\n");
		return -1;
	}
	
	/*allocate the memory for matrices*/
	inputMatrix1 = (cl_float*) malloc(sizeof(cl_float)*width*width);
	if(!inputMatrix1)
	{
		printf("memory allocation failed\n");
		ret = -1;
		goto CLEANUP;
	}
	
	inputMatrix2 = (cl_float*) malloc(sizeof(cl_float)*width*width);
	if(!inputMatrix2)
	{
		printf("memory allocation failed\n");
		ret = -1;
		goto CLEANUP;
	}
	
	result = (cl_float*) malloc(sizeof(cl_float)*width*width);
	if(!result)
	{
		printf("memory allocation failed\n");
		ret = -1;
		goto CLEANUP;
	}
	
	/*assign data to the input matrices*/
	for(y=0;y < width;y++)
	{
		for(x=0;x < width;x++)
		{
			inputMatrix1[y*width + x] = data;
			inputMatrix2[y*width + x] = data;
			result[y*width + x] = 0;
			data ++;
		}
	}

#ifdef DEBUG_MODE
printf ("The result from sequential mode is :\n");
int i,j,k;
i=0;j=0;k=0;
for(i=0;i < width;i++)
{
	for(j=0;j < width;j++)
	{
		result[i*width + j] = 0;
		
		for(k=0;k<width;k++)
		{
			result[i*width + j] += inputMatrix1[i*width + k] * inputMatrix2[k*width + j];
		}
		
		printf("%f\t",result[i*width + j]);
	}
	printf("\n");
}	


#endif /*DEBUG_MODE*/
	
	/* Context properties list (must be terminated with 0) */
	properties[0] = CL_CONTEXT_PLATFORM;
	properties[1] = (cl_context_properties) platform_id;
	properties[2] = 0;
	
	/* Create a context with the GPU device */
	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
	if (!context || err != CL_SUCCESS)
	{
       printf("clCreateContext failed\n");
       ret = -1;
	   goto CLEANUP;
	}

	/* Create a command queue using the context and device */
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE,&err);
	if (!command_queue || err != CL_SUCCESS)
	{
       printf("clCreateCommandQueue failed\n");
       ret = -1;
	   goto CLEANUP;
	}

	/* Load kernel file, prepend static info, and return total kernel size */
	kernelSource = loadProgSource("MatrixMul.cl", "", &kernelSize);
	if (!kernelSource)
	{
       printf("loadProgSource failed\n");
       ret = -1;
	   goto CLEANUP;
	}
	
#ifdef DEBUG_MODE
	printf("Kernel  = %s\n",kernelSource);
#endif /*DEBUG_MODE*/

	/* Create a program from the kernel source code */
	program = clCreateProgramWithSource(context, 1, (const char **) 
										&kernelSource, NULL, &err);
	if (!program || err != CL_SUCCESS)
	{
       printf("clCreateProgramWithSource failed\n");
       ret = -1;
	   goto CLEANUP;
	}
	
	/* Compile the program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		printf("Error building program\n");
		
		/*Print the compilation results*/
		char buffer[4096];
		size_t len;
		clGetProgramBuildInfo(program,device_id,CL_PROGRAM_BUILD_LOG,
						  sizeof(buffer),buffer,&len);
		printf("Build Log : %s\n",buffer);
	
		ret = -1;
		goto CLEANUP;
	}
	else
	{
		/*Print the compilation results*/
		char buffer[4096];
		size_t len;
		clGetProgramBuildInfo(program,device_id,CL_PROGRAM_BUILD_LOG,
						  sizeof(buffer),buffer,&len);
		if(strcmp(buffer,""))
			printf("Build Log : Compilation Successful\n");
		else
			printf("Build Log : %s\n",buffer);
	}
	
	/* Specify which kernel from the program to execute */
	kernel = clCreateKernel(program, "MatrixMul", &err);
	if (!kernel || err != CL_SUCCESS)
	{
       printf("clCreateKernel failed\n");
       ret = -1;
	   goto CLEANUP;
	}
	
	/* Create buffers for the input and output Matrices */
	inputMat1 = clCreateBuffer(context, CL_MEM_READ_ONLY, 
          sizeof(float) * width*width, NULL, NULL);
    inputMat2 = clCreateBuffer(context, CL_MEM_READ_ONLY, 
          sizeof(float) * width*width, NULL, NULL);
	outputMat = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
          sizeof(float) * width*width, NULL, NULL);
          
    
    /* Load data into the input buffers */
	clEnqueueWriteBuffer(command_queue, inputMat1, CL_TRUE, 0,
                       sizeof(float) * width*width, inputMatrix1, 0, NULL, NULL);
                       
	clEnqueueWriteBuffer(command_queue, inputMat2, CL_TRUE, 0,
                       sizeof(float) * width*width, inputMatrix2, 0, NULL, NULL);
                       
    /* Set the argument list for the kernel command */
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputMat1);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputMat2);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputMat);
	clSetKernelArg(kernel, 3, sizeof(int), &width);
	
	/*decompose the multiplication into small work-groups working in parallel*/
	size_t local[2], global[2];
	
	global[0] = width;
	global[1] = width;
	local[0] = BLOCK_SIZE;
	local[1] = BLOCK_SIZE;
	
	err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local,0, NULL, &prof_event);
    if (err != CL_SUCCESS)
    {
       printf("clEnqueueNDRangeKernel failed %d\n", err);
       ret = -1;
	   goto CLEANUP;
    }
	
	/*Synchronization Point*/
	err = clFinish(command_queue);
	if (err != CL_SUCCESS)
    {
       printf("clFinish failed %d\n", err);
       ret = -1;
	   goto CLEANUP;
    }
    
     // Copy the results from out of the output buffer
	clEnqueueReadBuffer(command_queue, outputMat, CL_TRUE, 0,
                      sizeof(float) * width*width, result, 0, NULL, NULL);
                      
    err = clWaitForEvents(1,&prof_event );
    if (err != CL_SUCCESS)
    {
       printf("clWaitForEvents failed %d\n", err);
       ret = -1;
	   goto CLEANUP;
    }
    
    double run_time;
    cl_ulong start_time, end_time;
	size_t return_bytes;
    
    err = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_START,
									sizeof(cl_ulong),&start_time,&return_bytes);
	if (err != CL_SUCCESS)
    {
       printf("clGetEventProfilingInfo failed %d\n", err);
       ret = -1;
	   goto CLEANUP;
    }
    
    err = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end_time,&return_bytes);
    if (err != CL_SUCCESS)
    {
       printf("clGetEventProfilingInfo failed %d\n", err);
       ret = -1;
	   goto CLEANUP;
    }
    
    run_time =(double)(end_time - start_time);
                      
     /*Print the result*/
     printf("The Kernel Execution Time for BLOCK_SIZE %d is %lf nanosecond\n",BLOCK_SIZE,run_time);
     
#ifdef DEBUG_MODE
     printf ("The kernel Product Matrix is:\n");
     for(y=0;y < width;y++)
	{
		for(x=0;x < width;x++)
		{
			printf("%f\t",result[y*width + x]);
		}
		printf("\n");
	}
#endif /*DEBUG_MODE*/	
	
CLEANUP:
	/*Deallocate the memory allocations*/
	if(inputMatrix1)
	{
		free(inputMatrix1);
		inputMatrix1 = NULL;
	}
	
	if(inputMatrix2)
	{
		free(inputMatrix2);
		inputMatrix2 = NULL;
	}
	
	if(result)
	{
		free(result);
		result = NULL;
	}
    
	/*release OpenCL resources*/
	if(context)
		clReleaseContext(context);
	if(command_queue)
		clReleaseCommandQueue(command_queue);
    if(program)
		clReleaseProgram(program);
	if(kernel)
		clReleaseKernel(kernel);
	if(inputMat1)
		clReleaseMemObject(inputMat1);
	if(inputMat2)
		clReleaseMemObject(inputMat2);
	if(outputMat)
		clReleaseMemObject(outputMat);
	
	return ret;
}


/*Function to determine GPU support*/
int CheckGPUSupport(cl_platform_id &platform_id,cl_device_id &device_id)
{
	cl_uint num_of_platforms = 0;
	cl_uint num_of_devices = 0;
	cl_int err;
	cl_device_id *mydevice;
	
	 // Retrive list of platforms available
	if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS) 
	{
		printf("Unable to get platform_id\n");
		return -1;
	}
	
	// Get a supported GPU device
	if (clGetDeviceIDs(platform_id,
						CL_DEVICE_TYPE_GPU, 1, 
						&device_id,&num_of_devices) != CL_SUCCESS) 
     {
		printf("Unable to get device_id\n");
		return -1;
	 }
	 
	// Create and load the device list:
	mydevice = new cl_device_id[num_of_devices];
	
	if (clGetDeviceIDs(platform_id,CL_DEVICE_TYPE_GPU,num_of_devices,mydevice,NULL)!= CL_SUCCESS)
	{
		printf("Unable to get device_id\n");
		return -1;
	}
	
	for (int i=0; i<num_of_devices; i++)
	{
		char buffer[10240];
		cl_uint buf_uint;
		cl_ulong buf_ulong;
		clGetDeviceInfo(mydevice[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
		printf("  The Device info : \n");
		printf("  DEVICE_NAME = %s\n", buffer);
		clGetDeviceInfo(mydevice[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
		printf("  DEVICE_VENDOR = %s\n", buffer);
		clGetDeviceInfo(mydevice[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
		printf("  DEVICE_VERSION = %s\n", buffer);
		clGetDeviceInfo(mydevice[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
		printf("  DRIVER_VERSION = %s\n", buffer);
		clGetDeviceInfo(mydevice[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
		printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
		clGetDeviceInfo(mydevice[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
		printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
		clGetDeviceInfo(mydevice[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
		printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
	}
	
	delete mydevice;
	
	return 0;
}

/*Function to read the offline kernel file*/
char* loadProgSource(const char* filename, const char* preamble, size_t *sz) 
{
  FILE* fptr = NULL;
  size_t szSource, szPreamble, howmany;
  char* sourceString;

  // Open the OpenCL source code file
  fptr = fopen(filename, "r");
  szPreamble = strlen(preamble);

  // Get the length of the source code
  fseek(fptr, 0, SEEK_END);
  szSource = ftell(fptr);
  fseek(fptr, 0, SEEK_SET);

  // Allocate a buffer for the source code string and read it in
  sourceString = (char *) calloc(szSource + szPreamble+1, sizeof(char));
  howmany = fread((sourceString) + szPreamble, szSource, 1, fptr);
  fclose(fptr);
  *sz = szSource + szPreamble;
  sourceString[szSource + szPreamble] = '\0';
  return sourceString;
}
