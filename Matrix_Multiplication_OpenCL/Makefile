########################################################################
#
#  DESCRIPTION  : Makefile to OpenCL application
# 
#  PLATFORM		: Linux
#
#  DATE	               	NAME	        	  	REASON
#  11th Nov,2018        Shashi Shivaraju        CPSC_6780_HW_03
#                       [C88650674]
########################################################################
#compiler
CC      = g++
C       = cpp

#auxiliary flags
CFLAGS  = -O

#set platform dependent variables
ifeq ("$(shell uname)", "Linux")
    LDFLAGS   = -L /usr/local/cuda/lib64/ -lX11 -lGL -lGLU -lglut -lm -lXmu -lOpenCL 
endif

INC = -I /usr/local/cuda/include

#executable name
PROJECT		= OpenCLMatMul

#list of all .o file for each .cpp file
OBJECTS = MatrixMul.o

#this does the linking step  
${PROJECT}:	MatrixMul.o
	${CC} ${CFLAGS} ${LFLAGS} -o ${PROJECT} ${OBJECTS} ${LDFLAGS} $(INC)

#compile the code 
%.o: %.cpp
	${CC} -c ${CFLAGS} $<

#this will clean up all temporary files created by make all
clean:
	rm -f core.* *.o *~ ${PROJECT}
