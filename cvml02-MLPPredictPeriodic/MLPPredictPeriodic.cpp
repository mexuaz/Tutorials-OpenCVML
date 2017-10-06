#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

template<typename type>
Mat_<type> sin(const Mat_<type>& x) {
	int channels = x.channels();
	int nRows = x.rows;
	int nCols = x.cols * channels;
	if (x.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}
	Mat_<type> y = x.clone();
	for(int i = 0; i < nRows; ++i)   {
		type* p = reinterpret_cast<type*>(y.ptr(i));
		for (int j = 0; j < nCols; ++j) {
			p[j] = sin(p[j]);
		}
	}
	return y;
}

template<typename type>
Mat_<type> cos(const Mat_<type>& x) {
	return sin<type>(M_PI/2 - x);
}


/*!
 * \brief linspace Return evenly spaced numbers within [start, stop) interval.
 * \param start Start of interval.
 * \param stop End of interval.
 * \param n Number of samples to generate.
 * \return Matrix of evenly spaced values in rows.
 */
template<typename type>
Mat_<type> linspace(type start, type stop, size_t n) {
    type step = (stop-start)/n;
    type* data = new type[n];
    if(!data) {
        cerr << "Memory allocation failed!" << endl;
        return Mat();
    }
    generate(data, data+n, [&]()->type {type t=start; start+=step;return t;});
    return Mat_<type>(int(n), 1, data);
}

/*!
 * \brief arange Return evenly spaced values within [start, stop) interval.
 * \param start Start of interval. The default start value is 0.
 * \param stop End of interval.
 * \param step Spacing between values.
 * \return Matrix of evenly spaced values in rows.
 */
template<typename type>
Mat_<type> arange(type start, type stop, type step) {
    size_t sz = size_t((stop-start)/step);
	type* data = new type[sz];
	if(!data) {
		cerr << "Memory allocation failed!" << endl;
		return Mat();
	}
    generate(data, data+sz, [&]()->type {type t=start; start+=step;return t;});
	return Mat_<type>(int(sz), 1, data);
}

template<typename type>
Mat_<type> arange(type stop, type step = static_cast<type>(1)) {
    return arange(static_cast<type>(0), stop, step);
}

template<typename type>
Mat_<type> fun(const Mat_<type>& x) {
    return sin<type>(x)+cos<type>(3*x);
}

Mat plot_results(const Mat& train_x, const Mat& train_y,
                 const Mat& test_x, const Mat& test_y) {
    // plot module only acceptes double values
    Mat xd, yd;
    train_x.convertTo(xd, CV_64F);
    train_y.convertTo(yd, CV_64F);

    Mat xtd, ytd;
    test_x.convertTo(xtd, CV_64F);
    test_y.convertTo(ytd, CV_64F);

    //adjust border and margins of the 2 plots to match  together
    double xmin, xmax;
    minMaxIdx(xd, &xmin, &xmax);
    double ymin, ymax;
    minMaxIdx(yd, &ymin, &ymax);

    double xtmin, xtmax;
    minMaxIdx(xtd, &xtmin, &xtmax);
    double ytmin, ytmax;
    minMaxIdx(ytd, &ytmin, &ytmax);

    xmin = min(xmin, xtmin)-1;
    xmax = max(xmax, xtmax)+1;
    ymin = min(ymin, ytmin)-1;
    ymax = max(ymax, ytmax)+1;

    Ptr<plot::Plot2d> plot_train = plot::Plot2d::create(xd, yd);
    plot_train->setMinX(xmin);
    plot_train->setMaxX(xmax);
    plot_train->setMinY(ymin);
    plot_train->setMaxY(ymax);
    plot_train->setPlotLineColor(Scalar(0, 255, 0)); // Green
    plot_train->setPlotLineWidth(3);
    plot_train->setNeedPlotLine(false);
    plot_train->setShowGrid(false);
    plot_train->setShowText(false);
    plot_train->setPlotAxisColor(Scalar(0, 0, 0)); // Black (invisible)
    Mat img_trainpl;
    plot_train->render(img_trainpl);


    Ptr<plot::Plot2d> plot_test = plot::Plot2d::create(xtd, ytd);
    plot_test->setMinX(xmin);
    plot_test->setMaxX(xmax);
    plot_test->setMinY(ymin);
    plot_test->setMaxY(ymax);
    plot_test->setShowGrid(false);
    plot_test->setShowText(false);
    Mat img_testpl;
    plot_test->render(img_testpl);

    // to conformt the common coordinates (x increase from bottom to top)
    Mat img;
    flip(img_trainpl+img_testpl, img, 0);

    return img;
}

int main() {
    // seeding random number generator of OpenCV
    cv::theRNG().state = uint64(cv::getTickCount());

    // OpenCV ml::ann module only uses float32 matrices
    // input traing data => number of rows: training samples
    // number of columns: data dimention=0-th (input) layer size
    Mat1f x(150, 1); // Or: Mat mat(150, 1, CV_32FC1);
    randu(x, Scalar(-1*M_PI), Scalar(+1*M_PI));
	
    // Initialize noise matrix of uniform distrbution
    Mat noise(x.size(), CV_32F);
    randu(noise, -.4, .4); // uniform distrbution
    //randn(noise, 0, .3); // normal distrbution

    // evaluate the function plus noise
    Mat y = fun(x) + noise;

	Ptr<ANN_MLP> net = ANN_MLP::create();
	
	Mat layerSizes = (Mat_<int>(5, 1) << x.cols, 4, 8, 4, y.cols);
	net->setLayerSizes(layerSizes);

    /*
     * sigmoid symmetric in OpenCV is 1.7159\*tanh(2/3 \* x),
     * same as tanh function in other libraries
     * The output will range from [-1.7159, 1.7159], instead of [0,1]
     */
	net->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM);
	TermCriteria termCrit = TermCriteria(
		TermCriteria::Type::COUNT + TermCriteria::Type::EPS,
		1e4, 1e-10);
	net->setTermCriteria(termCrit);
	net->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);
	
	Ptr<TrainData> trainingData = TrainData::create(
        x, SampleTypes::ROW_SAMPLE, y
	);
    // Spilits the traing set to have subset for testing
    trainingData->setTrainTestSplitRatio(.7, false); // shuffle already done

    cv::TickMeter t;
    // First will scale the input and output for training samples and
    // then Will use Nguyen-Widrow algorithm to initialize the weights
    t.start();
	net->train(trainingData);
    t.stop();
    double t_train = t.getTimeSec();
    t.reset();

    Mat resp;
    float rms = net->calcError(trainingData, true, resp);
	
    // plot perdicted results vs training samples to visualize the performance
	auto test = arange(-3.14f, +3.14f, .001f);
	Mat result;
    t.start();
	net->predict(test, result);
    t.stop();
    double t_predc = t.getTimeMilli();

    cout << "RMS: " << rms << endl
         << "Training Time: " << t_train << " s" << endl
         << "Prediction Time: " << t_predc << " ms" << endl;


    Mat img = plot_results(x, y, test, result);
    namedWindow("MLP");
    imshow("MLP", img);
	
	waitKey();
	return 0;
}
/*
BACKPROP
RMS: 0.762911
Training Time: 16.0031 s
Prediction Time: 1.7267 ms

RPROP
RMS: 1.08866
Training Time: 0.733092 s
Prediction Time: 1.65154 ms
*/
