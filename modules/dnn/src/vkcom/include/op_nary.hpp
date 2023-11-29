// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_OP_NARY_HPP
#define OPENCV_OP_NARY_HPP

#include "vkcom.hpp"
#include "op_base.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

enum NaryShaderType
{
    kNaryShaderTypeBinary,
    kNaryShaderTypeTrinary,
    kNaryShaderTypeNary,
    kNaryShaderTest,
};

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

    OpNary(const OPERATION naryOpType, int ninputs, int max_ndims, const int* shapesBuf, const size_t* stepsBuf);

    void firstForward(); // Execute only in the first forward.
    virtual bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) CV_OVERRIDE;
    Ptr<Tensor> weightTensorPtr;
private:
    bool binaryForward(std::vector<Tensor>& ins, std::vector<Tensor>& outs);
    bool trinaryForward(std::vector<Tensor>& ins, std::vector<Tensor>& outs);
    bool naryForward(std::vector<Tensor>& ins, std::vector<Tensor>& outs);

    const OPERATION naryOpType;
    NaryShaderType naryShaderType;
    int ninputs;
    int max_ndims;
    AutoBuffer<int32_t> shapesBuf;
    AutoBuffer<int32_t> stepsBuf;
    bool firstForwardFinsh = false;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
#endif //OPENCV_OP_MATMUL_HPP
