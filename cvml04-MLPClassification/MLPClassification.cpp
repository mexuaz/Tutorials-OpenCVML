#include <opencv2/opencv.hpp>
#include <opencv2/hdf.hpp>
#include <opencv2/plot.hpp>
#include <random>
#include <string>

using namespace std;

using namespace cv;
using namespace ml;

enum {
    XMIN,
    XMAX,
    YMIN,
    YMAX
};


/*!
 * \brief generate_samples_MVN Generates random 2D sets of Multi Variant Normal
 * distributions
 * \param dimention of samples
 * \param nSamples number of samples for each vector
 * \param nClasses number of classes or set of samples
 * \param vSamples vector containing samples
 * \param colors colors matrix
 */
void generate_samples_MVN(int dimention, int nSamples, int nClasses,
                          vector<Mat>& vSamples,
                          Mat& colors) {
    if(vSamples.size() != size_t(nClasses)) {
        vSamples.clear();
        vSamples.resize(size_t(nClasses));
    }

    // Matrix holding mean values of normal distributions
    Mat1f mean(nClasses, dimention);
    theRNG().fill(mean, RNG::UNIFORM, -200, 200);

    // Matrix holding covarience values for normal distributions
    Mat1f cov(nClasses, dimention*dimention);
    theRNG().fill(cov, RNG::UNIFORM, -30, 30);

    for(int i = 0; i < nClasses; i++) {

        Mat1f tmean = mean.row(i);
        Mat1f tcov = cov.row(i).reshape(0, 2);

        Mat samples;
        randMVNormal( tmean, tcov, nSamples, samples );
        vSamples[size_t(i)] = (samples);
    }

    // if colors matrix does not match the samples set size reassign it
    if(colors.rows < nClasses) {
        colors = Mat::zeros(nClasses, 1, CV_64FC3);
        // Randomize with light colors on black background
        theRNG().fill(colors, RNG::UNIFORM, 50, 255);
    }
}


/*!
 * \brief save_samples_hdf save samples matrices in hdf format for later use
 * \param filename
 * \param vSamples
 */
void save_samples_hdf(const string& filename,
                      const vector<Mat>& vSamples) {

    Ptr<hdf::HDF5> fhdf = hdf::open( filename );
    int i = 0;
    for(const auto& ms : vSamples) {
        fhdf->dswrite(ms, "dist"+to_string(i++));
    }
    fhdf->close();
    cout << "All samples are saved to " << filename << endl;
}

/*!
 * \brief load_samples_hdf load set of samples from a file and reassing the
 * color values if needed
 * \param filename
 * \param vSamples
 * \param colors
 */
void load_samples_hdf(const string& filename,
                      vector<Mat>& vSamples,
                      Mat& colors) {
    vSamples.clear();
    Ptr<hdf::HDF5> fhdf = hdf::open( filename );
    int i = 0;
    string label("dist"+to_string(i));
    while(fhdf->hlexists(label)) {
        Mat s;
        fhdf->dsread(s, label);
        vSamples.push_back(s);
        label = "dist"+to_string(++i);
    }
    fhdf->close();

    // if colors matrix does not match the samples set size reassign it
    if(colors.rows < int(vSamples.size())) {
        colors = Mat::zeros(int(vSamples.size()), 1, CV_64FC3);
        // Randomize with light colors on black background
        theRNG().fill(colors, RNG::UNIFORM, 50, 255);
    }


    cout << to_string(vSamples.size())
         << " set of samples are loaded." << endl;
}

/*!
 * \brief plot_samples plot all samples
 * \param vSamples vector of matrices containing samples
 * \param colors matrices of colors used for coloring samples
 * \param bounds boundries of image in real numbers
 * \return image of plotted samples
 */
Mat plot_samples(const vector<Mat>& vSamples,
                 const Mat& colors,
                 Vec4d& bounds) {


    // enough colors for every sample
    assert(colors.rows >= int(vSamples.size()));

    vector<Ptr<plot::Plot2d>> plots;

    bounds[XMIN] = bounds[YMIN] = DBL_MAX;
    bounds[XMAX] = bounds[YMAX] = DBL_MIN;

    for(size_t i = 0; i < vSamples.size(); i++) {

        // plot module only acceptes double values
        Mat samplesd;
        vSamples[i].convertTo(samplesd, CV_64F);
        //only 2 dimensions could be plotted
        assert(samplesd.cols == 2);

        Mat xd = samplesd.col(0);
        Mat yd = samplesd.col(1);

        double xmin, xmax;
        minMaxIdx(xd, &xmin, &xmax);
        double ymin, ymax;
        minMaxIdx(yd, &ymin, &ymax);

        bounds[XMIN] = min(bounds[XMIN], xmin);
        bounds[XMAX] = max(bounds[XMAX], xmax);
        bounds[YMIN] = min(bounds[YMIN], ymin);
        bounds[YMAX] = max(bounds[YMAX], ymax);


        Ptr<plot::Plot2d> plot = plot::Plot2d::create(xd, yd);

        plot->setPlotLineColor(colors.at<Vec3d>(int(i), 0));
        plot->setNeedPlotLine(false);
        plot->setShowGrid(false);
        plot->setShowText(false);
        plot->setPlotAxisColor(Scalar(0, 0, 0)); // Black (invisible)

        plots.push_back(plot);

    }

    // define a margin for better visualizing the sets
    double margin = 20;
    bounds[XMIN] -= margin;
    bounds[XMAX] += margin;
    bounds[YMIN] -= margin;
    bounds[YMAX] += margin;

    //adjust border and margins of all the plots to match  together
    Mat img;

    for(auto& plt : plots) {
        plt->setMinX(bounds[XMIN]);
        plt->setMaxX(bounds[XMAX]);
        plt->setMinY(bounds[YMIN]);
        plt->setMaxY(bounds[YMAX]);

        Mat img_plt;
        plt->render(img_plt);

        if(img.empty()) {
            img = img_plt.clone();
        } else {
            img += img_plt;
        }
    }

    return img;
}

/*!
 * \brief prepare_train_data perpare training sets given a vector of matrices
 * \param vSamples vector of matrices
 * \return training samples (unshuffle)
 */
static Ptr<TrainData> prepare_train_data(const vector<Mat>& xSamples) {

    // Generate output sample matrices for each
    vector<Mat> ySamples(xSamples.size());
    for(size_t i = 0; i <xSamples.size(); i++) {
        ySamples[i] = Mat(xSamples[i].rows, 1, xSamples[i].type(), Scalar(i));
    }
    Mat x, y;
    vconcat(xSamples, x);
    vconcat(ySamples, y);

    return TrainData::create(x, ROW_SAMPLE, y);
}


void plot_responses(const Ptr<ANN_MLP>& net,
                    const Vec4d& bounds,
                    Mat& img,
                    const Mat& colors,
                    int step = 20) {
    double xf = (bounds[XMAX] - bounds[XMIN])/img.cols;
    double yf = (bounds[YMAX] - bounds[YMIN])/img.rows;
    for(int c = 0; c < img.cols; c+=step) {
        for(int r = 0; r < img.rows; r+=step) {

            Mat1f pt = (Mat_<float>(1, 2) <<
                        float(xf * c + bounds[XMIN]),
                        float(yf * r + bounds[YMIN]));

            Mat res;
            net->predict(pt, res);
            int cat = cvRound(res.at<float>(0, 0));

            if(cat >= colors.rows) {
                cerr << "out of classes ranges "
                     << pt << " => " << cat << endl;
                continue;
            }

            cv::drawMarker(img, Point(c, r),
                           colors.at<Vec3d>(cat),
                           MarkerTypes::MARKER_TILTED_CROSS, step);
        }
    }
}

int main(int /*argc*/, char **/*argv*/) {
    theRNG().state = static_cast<uint64>(time(NULL));

    const int nClasses = 5; // number of classes
    const int nSamples = 300; // number of samples used in each class
    const int dim = 2; // each sample dimension (plot would fail > 2)

    Mat colors; // Matrix holding colors for classes
    vector<Mat> vSamples; // Matrix holding set of samples
    Vec4d bounds; // boudry values of image in real numbers
    Mat img; // image used for drawing sets
    int sw = 0; // switch used for detecting key press

    while(true) {
        if(sw == 27) { // ESC => exit the program
            break;
        } else if( sw == 's') { // save samples matrices for later use
            save_samples_hdf("samples.hdf5", vSamples);
        } else if(sw == 'l') { // load samples matrices from a file
            load_samples_hdf("samples.hdf5", vSamples, colors);
            img = plot_samples(vSamples, colors, bounds);
        } else if(sw == 'c') { // generate new set of random colors
            theRNG().fill(colors, RNG::UNIFORM, 50, 255);
            img = plot_samples(vSamples, colors, bounds);
        }  else if( sw == 'r') { // run nueral network on these matrices

            Ptr<TrainData> tdata = prepare_train_data(vSamples);
            tdata->setTrainTestSplitRatio(0.95, true); // no test set
            //tdata->shuffleTrainTest(); // only shuffle the training set

            // Create the network model
            Ptr<ANN_MLP> net = ANN_MLP::create();

            Mat1i layerSizes = (Mat_<int>(5, 1) <<
                                tdata->getSamples().cols, // input layer
                                4, 6, 4,
                                tdata->getResponses().cols); // output layer
            net->setLayerSizes(layerSizes);
            net->setActivationFunction(ANN_MLP::SIGMOID_SYM);
            net->setTermCriteria(
                        TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,
                                     1e4, DBL_EPSILON));
            net->setTrainMethod(ANN_MLP::RPROP, 0.001);

            cout << "Training ...";
            cout.flush();

            TickMeter t;
            t.start();
            net->train(tdata);
            t.stop();
            cout << " " << t.getTimeSec() << " (s)" << endl;


            float rms = net->calcError(tdata, true, noArray());
            cout << "RMS: " << rms << endl;

            t.reset();
            t.start();
            plot_responses(net, bounds, img, colors, 10);
            t.stop();
            cout << "Prediction: " << t.getTimeMilli() << " (ms)" << endl;

        } else { // Generates random 2D sets of Multi Variant Normal dists
            generate_samples_MVN(dim, nSamples, nClasses, vSamples, colors);
            img = plot_samples(vSamples, colors, bounds);
        }

        imshow("Samples", img);
        sw = waitKey();
    }

    return 0;
}
