# Compilers and linkers
CC=nvc++

# Compilation flags
CFLAGS= -O3 -std=c++11 -Wall

ifeq ($(OPENACC)$(useFMA),truefalse)
ACCFLAGS= -acc -gpu=cc60,cc70,managed,nofma -Minfo=all
ADDFLAGS=  -Mnofma
else ifeq ($(OPENACC)$(useFMA),truetrue)
ACCFLAGS= -acc -gpu=cc60,cc70,managed -Minfo=all
ADDFLAGS=
else
ACCFLAGS=
ADDFLAGS=
endif


# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o common.o functions.o matrix_mult.o 
# The binary to be built by this Makefile
BINARY= matrix_mult.exe


all: $(BINARY) 

$(BINARY): $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS) $(ADDFLAGS) $(ACCFLAGS) 

%.o: %.cpp Makefile
	$(CC) -c $(CFLAGS) $(ADDFLAGS) $(ACCFLAGS) $<

%.o: %.cc Makefile
	$(CC) -c $(CFLAGS) $(ADDFLAGS) $(ACCFLAGS) $<

clean:
	rm *.o $(BINARY)

