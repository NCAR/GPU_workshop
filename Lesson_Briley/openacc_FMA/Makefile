# Compilers and linkers
CC=nvc++

# Compilation flags
CFLAGS= -g -O3 -std=c++11 -Wall

ifeq ($(OPENACC),true)
ACCFLAGS= -acc -gpu=cc60,cc70,managed -Minfo=all
else 
ACCFLAGS= 
endif

# Linking Flags
LDFLAGS=

# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o common.o functions.o  
# The binary to be built by this Makefile
BINARY= fma.exe


all: $(BINARY) 

$(BINARY): $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS) $(ACCFLAGS) $(LDFLAGS) # $(CUDALDFLAGS) 

%.o: %.cpp Makefile
	$(CC) -c $(CFLAGS) $(ACCFLAGS) $<

clean:
	rm *.o fma.exe

