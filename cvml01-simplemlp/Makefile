CPP = g++
CXXFLAGS = -std=c++11 -O3
INC = -I/usr/local/opencv3.3-cpu/include
LIBS = -L/usr/local/opencv3.3-cpu/lib
LIBS += -lopencv_core -lopencv_ml

TARGETS = simplemlp

.DEFAULT: all

.PHONY: all debug clean

all: $(TARGETS)

debug: CXXFLAGS += -g

simplemlp: simplemlp.cpp
	$(CPP) $(CXXFLAGS) $(LIBS) $(INC) simplemlp.cpp -o simplemlp

clean:
	rm -f $(TARGETS) *.o
