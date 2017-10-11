TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += "/usr/local/opencv3.3-cpu/include"
LIBS += -L"/usr/local/opencv3.3-cpu/lib"

LIBS += -lopencv_core \
    -lopencv_highgui \
    -lopencv_ml \
    -lopencv_imgproc \
    -lopencv_hdf \
    -lopencv_plot




SOURCES += MLPClassification.cpp


