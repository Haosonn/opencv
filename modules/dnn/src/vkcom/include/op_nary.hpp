// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_OP_NARY_HPP
#define OPENCV_OP_NARY_HPP

#include "vkcom.hpp"
#include "op_base.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

struct NaryShaderConfig
{
    int local_size_x;
    int local_size_y;
    int local_size_z;
};


class OpNary : public OpBase
{
public:
    // Copied from nary_eltwise_layers.cpp
    enum class OPERATION 
    {
        AND = 0,
        EQUAL,
        GREATER,
        GREATER_EQUAL,
        LESS,
        LESS_EQUAL,
        OR,
        POW,
        XOR,
        BITSHIFT,
        MAX,
        MEAN,
        MIN,
        MOD,
        PROD,
        SUB,
        SUM,
        ADD,
        DIV,
        WHERE,
    };

    OpNary(const OPERATION naryOpType, int** shapes, size_t** steps);

    void firstForward(); // Execute only in the first forward.
    virtual bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) CV_OVERRIDE;
    Ptr<Tensor> weightTensorPtr;
private:
    const OPERATION naryOpType;
    bool firstForwardFinsh = false;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
#endif //OPENCV_OP_MATMUL_HPP
