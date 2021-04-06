# Compilers and linkers
CC=nvc++

# Compilation flags
CFLAGS= -O3 -std=c++11 -Wall

ACCFLAGS= -acc -gpu=cc60,cc70 -Minfo=accel -I${NVHPC_ROOT_PATH}/include

# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o common.o functions.o matrix_mult.o 
# The binary to be built by this Makefile
BINARY= matrix_mult.exe


all: $(BINARY) 

$(BINARY): $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS) $(ACCFLAGS) 

%.o: %.cpp Makefile
	$(CC) -c $(CFLAGS) $(ACCFLAGS) $<

%.o: %.cc Makefile
	$(CC) -c $(CFLAGS) $(ACCFLAGS) $<

clean:
	rm *.o $(BINARY)

