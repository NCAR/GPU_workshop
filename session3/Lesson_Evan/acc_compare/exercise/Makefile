# Compilers and linkers
CC=nvc++

# Compilation flags
CFLAGS= -g -O3 -std=c++11 -Wall

# Linking Flags
LDFLAGS=

ifeq ($(OPENACC),true)
ACCFLAGS= -acc -ta=tesla:cc70 -Minfo=all -I${NVHPC_ROOT_PATH}/include
else 
ACCFLAGS= 
endif

# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o functions.o common.o matmul.o
# The binary to be built by this Makefile
BINARY= matmul.exe


all: $(BINARY) 

$(BINARY): $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS) $(LDFLAGS) $(ACCFLAGS)

%.o: %.cpp Makefile
	$(CC) -c $(CFLAGS) $(ACCFLAGS) $<

%.o: %.cc Makefile
	$(CC) -c $(CFLAGS) $(ACCFLAGS) $<

clean:
	rm *.o $(BINARY)
