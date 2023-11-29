#include <opencv2/core.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main() {
    dnn::Net net = dnn::Net();
    net.setPreferableBackend(dnn::DNN_BACKEND_VKCOM);
    net.setPreferableTarget(dnn::DNN_TARGET_VULKAN);
    Mat input1 = Mat::ones(2, 2, CV_32F);
    Mat input2 = Mat::ones(2, 2, CV_32F);
    Mat output;
    dnn::LayerParams params = dnn::LayerParams();
    params.name = "NaryEltwise";
    params.type = "NaryEltwise";
    params.set("operation", "add");
    net.addLayer(params.name, params.type, params);
    net.setInputsNames({"input1", "input2"});
    net.setInput(input1, "input1");
    net.setInput(input2, "input2");
    net.connect(0, 0, 1, 0);
    net.connect(0, 1, 1, 1);
    vector<String> layerNames = net.getLayerNames();
    for (auto i : layerNames) {
        cout << "layer name: " << i << endl;
        auto v = net.getLayerInputs(net.getLayerId(i));
        cout << "  input layer: " << endl;
        for (auto j : v) {
            cout << "    " << j->name << endl;
        }
    }
    
    output = net.forward();
    cout << output << endl;
    return 0;
}
