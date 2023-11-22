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

OpNary::OpNary(const OpNary::OPERATION _naryOpType, int** _shapes, size_t** steps) : naryOpType(_naryOpType)
{
    //TODO(VK) 
    shader_name = "nary_eltwise_spv";
    switch(naryOpType) {
        case OPERATION::ADD:
            std::cout << "opencv/modules/dnn/src/vkcom/src/op_nary.cpp: VK nary-eltwise\n" << std::endl;
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
    CV_Assert((ins.size() == 1 || ins.size() == 2) && outs.size() == 1);



    
    Shape inputShape = ins[0].getShape();
    Shape outputShape = outs[0].getShape();
    CV_Assert(inputShape.size() == outputShape.size());

    CV_Assert(inputShape.size() == 2 || inputShape.size() == 4);

    firstForward();

    std::vector<int> param = {(int)naryOpType};

    std::vector<int> shape = {(int)param.size()};
    Tensor paramTensor = Tensor(reinterpret_cast<const char *>(param.data()), shape, kFormatInt32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    destTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // input
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // out
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
    };

    Ptr<Pipeline> pipeline = pipelineFactoryPtr->getPipeline(shader_name, destTypes);
    Ptr<Descriptor> desSet = pipeline->createSet();
    Ptr<CommandBuffer> cmdBuffer = cmdPoolPtr->allocBuffer();

    VkCommandBuffer cmdBufferReal = cmdBuffer->get();
    desSet->writeTensor(ins[0], 0);
    desSet->writeTensor(outs[0], 1);
    desSet->writeTensor(paramTensor, 2); // TODO(vk) change the parameter from pushconstance to buffer.

    cmdBuffer->beginRecord();
    pipeline->bind(cmdBufferReal, desSet->get());
    group_x_ = group_y_ = group_z_ = 1; 
    vkCmdDispatch(cmdBufferReal, group_x_, group_y_, group_z_);
    cmdBuffer->endRecord();

   
    cmdPoolPtr->submitAndWait(cmdBufferReal);

    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
