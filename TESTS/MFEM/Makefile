include ../../Makefile.in

RELATIVE_PATH = ../../
MFEM_PATH = ../../../mfem/build

INC = -I$(MFEM_PATH) -I$(RELATIVE_PATH)$(AATGS_PATH)/include $(INCLAPACKBLAS)
LIB_BLASLAPACK = $(LIBLAPACKBLAS)

LIB = -L$(MFEM_PATH) -lmfem -L$(RELATIVE_PATH)$(AATGS_PATH)/lib -laatgs $(LIBLAPACKBLAS) -lm

SRC = problem_mfem.o

ALLTESTS = test_mfem00.ex test_mfem02.ex test_mfem05.ex test_mfem10.ex test_mfem10_nltgcr.ex test_mfem19.ex test_mfem19_nltgcr.ex

default: $(ALLTESTS)
all: $(ALLTESTS)

%.o : %.cpp
	$(CXX) $(FLAGSCPP) $(INC) $(LIB) -o $@ -c $<

%.o : %.c
	$(CXX) $(FLAGS) $(INC) $(LIB) -o $@ -c $<

test_mfem00.ex: test_mfem00.o $(SRC)
	$(LINKXX) $(FLAGSCPP) $(INC) -o test_mfem00.ex test_mfem00.o $(SRC) $(LIB)

test_mfem02.ex: test_mfem02.o $(SRC)
	$(LINKXX) $(FLAGSCPP) $(INC) -o test_mfem02.ex test_mfem02.o $(SRC) $(LIB)

test_mfem05.ex: test_mfem05.o $(SRC)
	$(LINKXX) $(FLAGSCPP) $(INC) -o test_mfem05.ex test_mfem05.o $(SRC) $(LIB)

test_mfem10.ex: test_mfem10.o $(SRC)
	$(LINKXX) $(FLAGSCPP) $(INC) -o test_mfem10.ex test_mfem10.o $(SRC) $(LIB)

test_mfem10_nltgcr.ex: test_mfem10_nltgcr.o $(SRC)
	$(LINKXX) $(FLAGSCPP) $(INC) -o test_mfem10_nltgcr.ex test_mfem10_nltgcr.o $(SRC) $(LIB)

test_mfem19.ex: test_mfem19.o $(SRC)
	$(LINKXX) $(FLAGSCPP) $(INC) -o test_mfem19.ex test_mfem19.o $(SRC) $(LIB)

test_mfem19_nltgcr.ex: test_mfem19_nltgcr.o $(SRC)
	$(LINKXX) $(FLAGSCPP) $(INC) -o test_mfem19_nltgcr.ex test_mfem19_nltgcr.o $(SRC) $(LIB)

clean:
	rm -rf *.ex;rm -rf *.o;rm -rf *.a;rm -rf ./TempData;
