#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;

void printNetInfo(dnn::Net net) {
    vector<String> layerNames = net.getLayerNames();
    cout << "Input: " << net.getLayer(0)->name << endl;
    for (auto i : layerNames) {
        int id = net.getLayerId(i);
        cout << "layer name: " << i << ", id=" << id << endl;
        auto v = net.getLayerInputs(id);
        cout << "  input layer: " << endl;
        for (auto j : v) {
            cout << "    " << j->name << endl;
        }
    }
}

void printMatRec(const cv::Mat& mat, int depth, int *mat_dims, int *dims) {
    for (int i = 0; i < mat.dims - depth; i++) {
        cout << " ";
    }
    cout << "[";
    if (depth == 1) {
        for (int i = 0; i < mat_dims[0]; i++) {
            dims[0] = i;
            cout << mat.at<float>(dims) << ",";
        }
        dims[0] = 0;
    }
    else {
        cout << endl;
        for (int i = 0; i < mat_dims[depth - 1]; i++) {
            dims[depth - 1] = i;
            printMatRec(mat, depth - 1, mat_dims, dims);
        }
        for (int i = 0; i < mat.dims - depth; i++) {
            cout << " ";
        }
    }
    cout << "]" << endl;

}

void printMat(const cv::Mat& mat) {
    int ndims = mat.dims;
    vector<int> mat_dims;
    vector<int> dims;
    for (int i = 0; i < ndims; i++) {
        mat_dims.push_back(mat.size[i]);
        dims.push_back(0);
    }
    printMatRec(mat, ndims, mat_dims.data(), dims.data());
}


void cal(Mat &input1, Mat &input2, Mat &output, dnn::Net& net) {
    net.setInput(input1, "input1");
    net.setInput(input2, "input2");

    cout << "\033[93m" << "First forwarding." << "\033[0m\n";
    output = net.forward();
    cout << "\033[93m" << "Formal forwarding." << "\033[0m\n";

    auto begin = std::chrono::high_resolution_clock::now();
    output = net.forward();
    auto end = std::chrono::high_resolution_clock::now();
    cout << "\033[93mTime elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\033[0m\n";
}

Mat testSingle(Mat &input1, Mat &input2, dnn::Net& net, bool useVK, bool print = false, bool useInternelPrint = true)
{
    Mat output;
    if (useVK)
    {
        net.setPreferableBackend(dnn::DNN_BACKEND_VKCOM);
        net.setPreferableTarget(dnn::DNN_TARGET_VULKAN);
        cout << "\033[95mUsing Vulkan.\033[0m\n";
    }
    else
    {
        net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(dnn::DNN_TARGET_CPU);
        cout << "\033[95mUsing CPU.\033[0m\n";
    }

    cal(input1, input2, output, net);
    if (print)
    {
        cout << "input1: " << endl;
        if (useInternelPrint)
            cout << input1 << endl;
        else
            printMat(input1);
        cout << "input2: " << endl;
        if (useInternelPrint)
            cout << input2 << endl;
        else
            printMat(input2);

        cout << "output: " << endl;
        if (useInternelPrint)
            cout << output << endl;
        else
            printMat(output);
    }
    return output;
}

void verifyResult(Mat& mat1, Mat& mat2)
{
    size_t sz = 0;
    for (auto it1 = mat1.begin<float>(), it2 = mat2.begin<float>(); it1 != mat1.end<float>(); ++it1, ++it2, ++sz)
    {
        if (std::fabs(*it1 - *it2) > 1e-9)
        {
            cout << "\033[91mElement unmatch: " << *it1 << " != " << *it2 << ", at " << sz << "\033[0m\n";
            abort();
            //return;
        }
    }
    cout << "\033[92mResults passed verification.\033[0m\n";
}

void validityTest(dnn::Net& net)
{
    Mat input1, input2;

    input1 = Mat::ones(8, 8, CV_32F);
    input2 = Mat::ones(8, 1, CV_32F);
    input1.at<float>(3, 2) = 25;
    input2.at<float>(3, 0) = 17;

    Mat output1 = testSingle(input1, input2, net, true, true);
    Mat output2 = testSingle(input1, input2, net, false, true);
    verifyResult(output1, output2);
}

void speedTest(dnn::Net &net)
{
    Mat input1, input2;

    int matDimH = 4096, matDimW = 4096;
    input1 = Mat::ones(matDimH, matDimW, CV_32F);
    input2 = Mat::ones(matDimH, matDimW, CV_32F);
    input1.at<float>(2000, 723) = 608.53f;
    input1.at<float>(712, 4) = 123.1231f;
    input1.at<float>(38, 218) = 21.12f;
    input1.at<float>(64, 213) = 2813.2f;
    input1.at<float>(2000, 0) = 213.231f;
    input2.at<float>(64, 723) = -27.0f;
    input2.at<float>(128, 4) = 18.5f;
    input2.at<float>(256, 218) = -212223.9f;
    input2.at<float>(2000, 0) = -1.7e4f;
    input2.at<float>(274, 0) = 2.3e5f;

    Mat output1 = testSingle(input1, input2, net, true, false);
    Mat output2 = testSingle(input1, input2, net, false, false);
    verifyResult(output1, output2);
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_DEBUG);
    dnn::Net net = dnn::Net();
    net.setPreferableBackend(dnn::DNN_BACKEND_VKCOM);
    net.setPreferableTarget(dnn::DNN_TARGET_VULKAN);
    dnn::LayerParams params = dnn::LayerParams();
    params.name = "NaryEltwise";
    params.type = "NaryEltwise";
    params.set("operation", "add");
    net.addLayer(params.name, params.type, params);
    net.setInputsNames({"input1", "input2"});
    net.connect(0, 0, 1, 0);
    net.connect(0, 1, 1, 1);

    //validityTest(net);
    speedTest(net);
    return 0;
}
