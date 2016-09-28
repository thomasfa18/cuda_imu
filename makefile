.SUFFIXES:             # Delete default suffixes
.SUFFIXES: .cu .o .h   # Define suffix list

COMPILER = nvcc
CFLAGS = -lm -I /usr/local/cuda-7.5/samples/common/inc
COBJS =  file_handling.o math_functions.o
CEXES = imu
all: ${CEXES}

.cu.o:
	${COMPILER} ${CFLAGS} $< -c

imu: imu.cu ${COBJS}
	${COMPILER} ${CFLAGS} imu.cu ${COBJS} -o imu

clean:
	rm -f *.o *~ ${CEXES}
