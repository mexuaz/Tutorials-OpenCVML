#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main() {

    // y = (x0 XOR x1) AND x2
    Mat in = (Mat_<float>(8, 3) <<
                   0.f, 0.f, 0.f,
                   0.f, 0.f, 1.f,
                   0.f, 1.f, 0.f,
                   0.f, 1.f, 1.f,
                   1.f, 0.f, 0.f,
                   1.f, 0.f, 1.f,
                   1.f, 1.f, 0.f,
                   1.f, 1.f, 1.f);

    Mat out = (Mat_<float>(8, 1) <<
                    0.f,
                    1.f,
                    0.f,
                    0.f,
                    0.f,
                    0.f,
                    0.f,
                    1.f);

    Ptr<ANN_MLP> net = ANN_MLP::create();

    // 3 is the hidden layer size
    Mat layerSizes = (Mat_<int>(3, 1) << in.cols, 3, out.cols);
    net->setLayerSizes(layerSizes);

    net->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM);

    TermCriteria termCrit = TermCriteria(
        TermCriteria::Type::COUNT + TermCriteria::Type::EPS,
        1e5, 1e-15);
    net->setTermCriteria(termCrit);

    net->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);

    Ptr<TrainData> trainingData = TrainData::create(
        in.rowRange(0, 6),
        SampleTypes::ROW_SAMPLE,
        out.rowRange(0, 6)
    );

    net->train(trainingData);

    Mat result;
    net->predict(in, result);

    cout << result << endl;

    return 0;
}
