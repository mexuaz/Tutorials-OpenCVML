CPP = g++
CXXFLAGS = -std=c++11 -O3
INC = -I/usr/local/opencv3.3-cpu/include
LIBS = -L/usr/local/opencv3.3-cpu/lib
LIBS += -lopencv_core \
	-lopencv_highgui \
	-lopencv_ml \
	-lopencv_hdf \
	-lopencv_plot \
	-lopencv_imgproc

TARGETS = MLPClassification
.DEFAULT: all
.PHONY: all debug clean
all: $(TARGETS)
debug: CXXFLAGS += -g
MLPClassification: MLPClassification.cpp
	$(CPP) $(CXXFLAGS) $(LIBS) $(INC) MLPClassification.cpp -o MLPPredictPeriodic
clean:
	rm -f $(TARGETS) *.o