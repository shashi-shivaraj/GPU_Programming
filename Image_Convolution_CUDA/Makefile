########################################################################
#
#  DESCRIPTION  : Makefile to compile CUDA convolution application
# 
#  PLATFORM		: Linux
#
#  DATE	               	NAME	        	  	REASON
#  5th Dec,2018         Shashi Shivaraju        CPSC_6780_Final_Project
#                       [C88650674]
########################################################################
#compiler
CC      = nvcc -g
C       = cu

#executable name
PROJECT		= convolution

#list of all .o file for each .cpp file
OBJECTS = ${PROJECT}.o

#this does the linking step  
${PROJECT}:	${OBJECTS}
	${CC} -o ${PROJECT} ${OBJECTS}

#compile the code 
%.o: %.cu
	${CC} -c ${CFLAGS} $<

#this will clean up all temporary files created by make all
clean:
	rm -f core.* *.o *~ ${PROJECT}
