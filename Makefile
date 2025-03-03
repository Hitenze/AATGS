include Makefile.in

SRC = ./SRC/utils/utils.o\
	./SRC/utils/memory.o\
	./SRC/utils/protos.o\
	./SRC/ops/vecops.o\
	./SRC/ops/matops.o\
	./SRC/problems/problem.o\
	./SRC/problems/bratu.o\
	./SRC/problems/lennardjones.o\
	./SRC/problems/hequation.o\
	./SRC/optimizer/optimizer.o\
	./SRC/optimizer/gd.o\
	./SRC/optimizer/adam.o\
	./SRC/optimizer/anderson.o\
	./SRC/optimizer/nltgcr.o

RELATIVE_PATH = ./

INC = -I./INC $(INCLAPACKBLAS)
LIB_BLASLAPACK = $(LIBLAPACKBLAS)

LIB = -L$(RELATIVE_PATH)$(AATGS_PATH)/lib -laatgs $(LIBLAPACKBLAS) -lm

SHELL := /bin/bash
ifeq ($(SHELL),/bin/zsh)
    SOURCE := source
else
    SOURCE := .
endif

default: libaatgs.a
all: libaatgs.a
lib: libaatgs.a

%.o : %.c
	$(CC) $(FLAGS) $(INC) -o $@ -c $<

libaatgs.a: $(SRC)
	$(AR) $@ $(SRC)
	$(RANLIB) $@
	rm -rf build;mkdir build;mkdir build/lib;mkdir build/include;
	cd SRC/utils;$(SOURCE) header.sh;cd ../..;
	cd SRC/ops;$(SOURCE) header.sh;cd ../..;
	cd SRC/problems;$(SOURCE) header.sh;cd ../..;
	cd SRC/optimizer;$(SOURCE) header.sh;cd ../..;
	cp libaatgs.a build/lib;cp INC/*.h build/include;
	$(CC) -shared $(FLAGS) $(INC) -o libaatgs.so $(SRC) $(LIB)
	cp libaatgs.so build/lib;
clean:
	rm -rf ./SRC/*.o;rm -rf ./SRC/utils/*.o;rm -rf ./SRC/ops/*.o;rm -rf ./SRC/problems/*.o;rm -rf ./SRC/optimizer/*.o;rm -rf ./EXTERNAL/*.o;rm -rf ./build;rm -rf *.a;rm -rf *.so;
