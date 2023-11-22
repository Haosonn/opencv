// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "internal.hpp"
#include "../include/op_nary.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

#define KSTRIP_LEN 32
#define BLOCK_SIZE 64

OpNary::OpNary(const OpNary::OPERATION _naryOpType, int _ninputs, int _max_ndims, int* _shapesBuf, size_t* _stepsBuf) : naryOpType(_naryOpType), ninputs(_ninputs), max_ndims(_max_ndims)
{
    shapesBuf.resize((ninputs + 1) * max_ndims);
    std::copy(_shapesBuf, _shapesBuf + (ninputs + 1) * max_ndims, shapesBuf.data());
    stepsBuf.resize((ninputs + 1) * max_ndims);
    std::copy(_stepsBuf, _stepsBuf + (ninputs + 1) * max_ndims, stepsBuf.data());

    //TODO(VK) 
    shader_name = "nary_eltwise_spv";
    switch(naryOpType) {
        case OPERATION::ADD:
            std::cout << "opencv/modules/dnn/src/vkcom/src/op_nary.cpp: on op ADD, VK nary-eltwise\n" << std::endl;
            break;
        //TODO(VK) add other cases
        default:
            CV_Error(Error::StsNotImplemented, "Unsupported nary operation type");
    }
    // TODO(VK): initialize OpNary class
}

void OpNary::firstForward()
{
    // TODO(VK) initialize first forward
    if (!firstForwardFinsh)
    {
        firstForwardFinsh = true;
    }
    else
        return;
}

bool OpNary::forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs)
{
    // TODO(VK): CV_assert for necessary conditions

    firstForward();

    std::vector<int> param = {(int)naryOpType, ninputs, max_ndims};
    std::vector<int> paramSize = {(int)param.size()};
    std::vector<int> ndims = {max_ndims};
    Tensor paramTensor = Tensor(reinterpret_cast<const char *>(param.data()), paramSize, kFormatInt32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    Tensor shapeTensor = Tensor(reinterpret_cast<const char *>(shapesBuf.data()), ndims, kFormatInt32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    Tensor stepTensor = Tensor(reinterpret_cast<const char *>(stepsBuf.data()), ndims, kFormatInt32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    destTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // input1
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // input2
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // out
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // param
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // shape
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // step
    };


    Ptr<Pipeline> pipeline = pipelineFactoryPtr->getPipeline(shader_name, destTypes);
    Ptr<CommandBuffer> cmdBuffer = cmdPoolPtr->allocBuffer();
    Ptr<Descriptor> desSet = pipeline->createSet();
    group_x_ = group_y_ = group_z_ = 1; 
    VkCommandBuffer cmdBufferReal = cmdBuffer->get();
    desSet->writeTensor(paramTensor, 3);
    desSet->writeTensor(shapeTensor, 4);
    desSet->writeTensor(stepTensor, 5);


    for (int i = 1; i <= ninputs - 1; i++)
    {
        desSet->writeTensor(i == 1 ? ins[0] : outs[0], 0);
        desSet->writeTensor(ins[i], 1);
        desSet->writeTensor(outs[0], 2); //TODO(VK): is it ok to write the same argument in both input and output?

        cmdBuffer->beginRecord();
        pipeline->bind(cmdBufferReal, desSet->get());
        vkCmdDispatch(cmdBufferReal, group_x_, group_y_, group_z_);
        cmdBuffer->endRecord();
        cmdPoolPtr->submitAndWait(cmdBufferReal);
    }

    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
