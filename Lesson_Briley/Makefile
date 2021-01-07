CC=pgc++
CUDACC=nvcc
LDFLAGS=
CFLAGS= -g -O3 -std=c++11 
CUDALIB=/glade/u/apps/dav/opt/cuda/11.0.3
CUDAFLAGS= -g -std=c++11 -O3 

all: fma.exe

fma.exe: main.o FMA.o functions.o 
	$(CC) $(CFLAGS) $(LDFLAGS) -lcudart -lcublas -lrt -ldl -I${CUDALIB}/include/ -L${CUDALIB}/lib64/  -o fma.exe FMA.o main.o functions.o

%.o: %.cpp
	$(CC) -c $(CFLAGS) $<

%.o: %.cu
	$(CUDACC) -c $(CUDAFLAGS) -I${CUDALIB}/lib64/ -L${CUDALIB}/include/ $<

clean:
	rm *.o fma.exe

run:
	./fma.exe
