include ../Makefile.in

RELATIVE_PATH = ../

INC = -I$(RELATIVE_PATH)$(AATGS_PATH)/include $(INCLAPACKBLAS)
LIB_BLASLAPACK = $(LIBLAPACKBLAS)

LIB = -L$(RELATIVE_PATH)$(AATGS_PATH)/lib -laatgs $(LIBLAPACKBLAS) -lm

SRC = 

ALLTESTS = test_bratu.ex test_LJ.ex test_hequ.ex

default: $(ALLTESTS)
all: $(ALLTESTS)

%.o : %.cpp
	$(CXX) $(FLAGSCPP) $(INC) -o $@ -c $<

%.o : %.c
	$(CC) $(FLAGS) $(INC) -o $@ -c $<

test_bratu.ex: test_bratu.o $(SRC)
	$(LINK) $(FLAGS) $(INC) -o test_bratu.ex test_bratu.o $(SRC) $(LIB)

test_LJ.ex: test_LJ.o $(SRC)
	$(LINK) $(FLAGS) $(INC) -o test_LJ.ex test_LJ.o $(SRC) $(LIB)

test_hequ.ex: test_hequ.o $(SRC)
	$(LINK) $(FLAGS) $(INC) -o test_hequ.ex test_hequ.o $(SRC) $(LIB)

clean:
	rm -rf *.ex;rm -rf *.o;rm -rf *.a;rm -rf ./TempData;
