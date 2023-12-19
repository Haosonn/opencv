// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "internal.hpp"
#include "../include/op_nary.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

#define STEP_SIZE 65536

#define MAX_GROUP_COUNT_X 65535
#define MAX_GROUP_COUNT_Y 65535
#define MAX_GROUP_COUNT_Z 65535

OpNary::OpNary(const OpNary::OPERATION _naryOpType, int _ninputs, int _max_ndims, const int* _shapesBuf, const size_t* _stepsBuf) : naryOpType(_naryOpType), ninputs(_ninputs), max_ndims(_max_ndims)
{
    CV_Assert(ninputs > 1);

    shapesBuf.resize((ninputs + 1) * max_ndims);
    std::transform(_shapesBuf, _shapesBuf + (ninputs + 1) * max_ndims, shapesBuf.data(), [](size_t x) { return static_cast<int32_t>(x); });
    stepsBuf.resize((ninputs + 1) * max_ndims);
    std::transform(_stepsBuf, _stepsBuf + (ninputs + 1) * max_ndims, stepsBuf.data(), [](size_t x) { return static_cast<int32_t>(x); });


    // TODO(VK): support more types of operation
    switch(naryOpType) {
        // case OPERATION::EQUAL:
        // case OPERATION::GREATER:
        // case OPERATION::GREATER_EQUAL:
        // case OPERATION::LESS:
        // case OPERATION::LESS_EQUAL:
        // case OPERATION::POW:
        // case OPERATION::BITSHIFT:
        // case OPERATION::MOD:
        // case OPERATION::PROD:
        // case OPERATION::SUB:
        case OPERATION::ADD:
        // case OPERATION::DIV:
        // case OPERATION::AND:
        // case OPERATION::OR:
        // case OPERATION::XOR:
        {
            CV_Assert(ninputs == 2);
            CV_Assert(max_ndims >= 2);
            shaderType = kNaryShaderTypeBinary;
            shader_name = "nary_eltwise_binary_forward_spv";

            // TODO(VK): confirm if this makes any sense
            nplanes = std::accumulate(shapesBuf.data(), shapesBuf.data() + max_ndims - 2, 1, [](int32_t a, int32_t b) { return a * b; } );
            N2 = shapesBuf.data()[max_ndims - 2];
            N1 = shapesBuf.data()[max_ndims - 1];
            CV_LOG_DEBUG(NULL, "max_ndims="<<max_ndims<<", nplanes="<<nplanes<<", N2="<<N2<<", N1="<<N1);
            break;
        }
        case OPERATION::WHERE:
        {
            CV_Assert(ninputs == 3);
            CV_Assert(max_ndims >= 2);
            shaderType = kNaryShaderTypeTrinary;
            shader_name = "nary_eltwise_trinary_forward_spv";
            break;
        }
        // case OPERATION::MAX:
        // case OPERATION::MEAN:
        // case OPERATION::MIN:
        case OPERATION::SUM:
        {
            CV_Assert(max_ndims >= 2);
            shaderType = kNaryShaderTypeNary;
            shader_name = "nary_eltwise_nary_forward_spv";
            break;
        }
        //TODO(VK) add other cases
        default:
            CV_Error(Error::StsNotImplemented, "Unsupported nary operation type");
    }
    // TODO(VK): initialize OpNary class
}

void OpNary::firstForward()
{
    if (!firstForwardFinsh)
    {
        config.local_size_x = 1; // TODO(vk) determine local_size_y if necessary
        config.local_size_y = 1; // TODO(vk) determine local_size_y if necessary
        config.local_size_z = 1; // TODO(vk) determine local_size_z if necessary
        computeGroupCount();
        firstForwardFinsh = true;
    }
    else
        return;
}

bool OpNary::binaryForward(std::vector<Tensor>& ins, std::vector<Tensor>& outs)
{
    std::vector<int32_t> param = {(int32_t)naryOpType, max_ndims};
    std::vector<int32_t> paramSize = {(int32_t)param.size()};
    std::vector<int32_t> dimSizes = {(ninputs + 1) * max_ndims};
    std::vector<int32_t> actualSteps;

    // TODO(VK): compute step for different dtype. Currently this is for kFormatFp32.
    actualSteps.resize(stepsBuf.size());
    std::transform(stepsBuf.data(), stepsBuf.data() + dimSizes[0], actualSteps.begin(), [](int32_t sz){ return sz / 4; });

    Tensor paramTensor = Tensor(reinterpret_cast<const char *>(param.data()), paramSize, kFormatInt32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    Tensor shapeTensor = Tensor(reinterpret_cast<const char *>(shapesBuf.data()), dimSizes, kFormatInt32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Tensor stepTensor = Tensor(reinterpret_cast<const char *>(actualSteps.data()), dimSizes, kFormatInt32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    destTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // input1
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // input2
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // out
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // param
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // shape
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // step
    };


    Ptr<Pipeline> pipeline = pipelineFactoryPtr->getPipeline(shader_name, destTypes);
    Ptr<CommandBuffer> cmdBuffer = cmdPoolPtr->allocBuffer();
    Ptr<Descriptor> desSet = pipeline->createSet();
    VkCommandBuffer cmdBufferReal = cmdBuffer->get();

    // TODO: remove experimental time counter
    auto begin = std::chrono::high_resolution_clock::now();

    desSet->writeTensor(ins[0], 0);
    desSet->writeTensor(ins[1], 1);
    desSet->writeTensor(outs[0], 2);
    desSet->writeTensor(paramTensor, 3);
    desSet->writeTensor(shapeTensor, 4);
    desSet->writeTensor(stepTensor, 5);

    // TODO: remove experimental time counter
    auto end = std::chrono::high_resolution_clock::now();
    CV_LOG_INFO(NULL, "Time elapsed to writeTensor: "<<(int)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<" ms");

    // TODO: remove experimental time counter
    begin = std::chrono::high_resolution_clock::now();

    cmdBuffer->beginRecord();
    pipeline->bind(cmdBufferReal, desSet->get());
    vkCmdDispatch(cmdBufferReal, group_x_, group_y_, group_z_);
    cmdBuffer->endRecord();
    cmdPoolPtr->submitAndWait(cmdBufferReal);

    // TODO(VK): remove experimental time counter
    end = std::chrono::high_resolution_clock::now();
    CV_LOG_INFO(NULL, "Time elapsed to compute binary forward: "<<(int)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<" ms");

    return true;
}

bool OpNary::forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs)
{
    // TODO(VK): CV_assert for necessary conditions

    firstForward();

    // TODO(VK): Support more dtypes. Currently only kFormatFp32 is supported.
    for (auto &tensor: ins)
    {
        CV_Assert(tensor.getFormat() == kFormatFp32);
    }
    for (auto &tensor: outs)
    {
        CV_Assert(tensor.getFormat() == kFormatFp32);
    }

    switch(shaderType) {
        case kNaryShaderTypeBinary: {
            // std::cout << "Dispatched binary operation.\n"; // TODO(VK): delete this
            return binaryForward(ins, outs);
            break;
        }
        default:
            CV_Error(Error::StsNotImplemented, "Unsupported shader type invoked.");
    }

    return true;
}

bool OpNary::computeGroupCount()
{
    if (shaderType == kNaryShaderTypeBinary)
    {
        group_x_ = nplanes; // parallelism at plane level
        group_y_ = N2;
        group_z_ = 1;
        // group_y_ = alignSize(N2, STEP_SIZE) / STEP_SIZE; // TODO(VK): Experimental batched opearation
        // group_z_ = alignSize(N1, STEP_SIZE) / STEP_SIZE; // TODO(VK): Experimental batched opearation
    }
    else
    {
        CV_Error(CV_StsNotImplemented, "shader type is not supported at compute GroupCount.");
    }

    CV_Assert(group_x_ <= MAX_GROUP_COUNT_X);
    CV_Assert(group_y_ <= MAX_GROUP_COUNT_Y);
    CV_Assert(group_z_ <= MAX_GROUP_COUNT_Z);

    // TODO(VK): delete this
    CV_LOG_DEBUG(NULL, "dispatching: group_x_="<<group_x_<<", group_y_="<<group_y_<<", group_z_="<<group_z_);

    return true;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
