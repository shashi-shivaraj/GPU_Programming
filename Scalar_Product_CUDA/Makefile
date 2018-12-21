########################################################################
#
#  DESCRIPTION  : Makefile to compile CUDA program
# 
#  PLATFORM		: Linux
#
#  DATE	               	NAME	        	  	REASON
#  11th Oct,2018         Shashi Shivaraju       CPSC_6780_Assignemnt_2
#                       [C88650674]
########################################################################
#compiler
CC      = nvcc -g
C       = cu

#executable name
PROJECT		= big_dot

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
