CPP = g++
CXXFLAGS = -std=c++11 -O3
INC = -I/usr/local/opencv3.3-cpu/include
LIBS = -L/usr/local/opencv3.3-cpu/lib
LIBS += -lopencv_core -lopencv_highgui -lopencv_ml -lopencv_plot
TARGETS = MLPPredictPeriodic
.DEFAULT: all
.PHONY: all debug clean
all: $(TARGETS)
debug: CXXFLAGS += -g
MLPPredictPeriodic: MLPPredictPeriodic.cpp
	$(CPP) $(CXXFLAGS) $(LIBS) $(INC) MLPPredictPeriodic.cpp -o MLPPredictPeriodic
clean:
	rm -f $(TARGETS) *.o