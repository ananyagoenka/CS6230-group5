CXX = /opt/homebrew/opt/llvm/bin/clang++
CC = /opt/homebrew/opt/llvm/bin/clang
K?=51
LOG?=2
D?=0
PRINT_UPPERBOUND?=10000
COMPILE_TIME_PARAMETERS=-DKMER_SIZE=$(K) -DLOG_LEVEL=$(LOG) -DDEBUG=$(D) -DUPPERBOUND=$(PRINT_UPPERBOUND)
OPT=

ifeq ($(D), 1)
OPT+=-g -O2 -fsanitize=address -fno-omit-frame-pointer
else
OPT+=-O3
endif

MPI_INC = -I/opt/homebrew/opt/open-mpi/include
FLAGS=$(OPT) $(COMPILE_TIME_PARAMETERS) -DTHREADED -Xpreprocessor -fopenmp -Wall -std=c++17 -I./include -I./src $(MPI_INC)

COMPILER=$(CXX)

LDFLAGS=-L/opt/homebrew/opt/llvm/lib -lomp -L/opt/homebrew/opt/open-mpi/lib -lmpi

OBJECTS=obj/timer.o \
		obj/dnaseq.o \
		obj/dnabuffer.o \
		obj/fastaindex.o \
		obj/hashfuncs.o \
		obj/kmerops.o \

all: kcount

install: kcount
	cp kcount $(HOME)/bin

kcount: obj/main.o $(OBJECTS)
	$(COMPILER) $(FLAGS) -o $@ $^ $(LDFLAGS)

obj/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(COMPILER) $(FLAGS) -c -o $@ $<

obj/main.o: src/main.cpp include/timer.hpp include/dnaseq.hpp include/dnabuffer.hpp include/fastaindex.hpp include/kmerops.hpp
obj/timer.o: src/timer.cpp include/timer.hpp
obj/dnaseq.o: src/dnaseq.cpp include/dnaseq.hpp
obj/dnabuffer.o: src/dnabuffer.cpp include/dnabuffer.hpp include/dnaseq.hpp
obj/fastaindex.o: src/fastaindex.cpp include/fastaindex.hpp include/dnaseq.hpp include/dnabuffer.hpp
obj/hashfuncs.o: src/hashfuncs.cpp include/hashfuncs.hpp
obj/kmerops.o: src/kmerops.cpp include/kmerops.hpp include/kmer.hpp include/dnaseq.hpp include/timer.hpp include/dnabuffer.hpp

clean:
	rm -rf *.o obj/* kcount $(HOME)/bin/kcount