// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

namespace cv { namespace dnn { namespace vkcom {

extern const unsigned int nary_eltwise_spv[660] = {
    0x07230203,0x00010000,0x0008000b,0x00000071,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0007000f,0x00000005,0x00000004,0x6e69616d,0x00000000,0x0000000e,0x0000001d,0x00060010,
    0x00000004,0x00000011,0x00000100,0x00000001,0x00000001,0x00030003,0x00000002,0x000001c2,
    0x00040005,0x00000004,0x6e69616d,0x00000000,0x00050005,0x00000006,0x7972616e,0x6464615f,
    0x00000028,0x00040005,0x0000000a,0x646e496d,0x00007865,0x00060005,0x0000000e,0x575f6c67,
    0x476b726f,0x70756f72,0x00004449,0x00040005,0x00000016,0x646e496e,0x00007865,0x00040005,
    0x0000001c,0x61636f6c,0x00785f6c,0x00080005,0x0000001d,0x4c5f6c67,0x6c61636f,0x6f766e49,
    0x69746163,0x44496e6f,0x00000000,0x00040005,0x00000023,0x61636f6c,0x00795f6c,0x00050005,
    0x00000028,0x6f6c5f61,0x5f6c6163,0x00000078,0x00050005,0x0000002e,0x6f6c5f61,0x5f6c6163,
    0x00000079,0x00050005,0x00000033,0x6f6c5f62,0x5f6c6163,0x00000078,0x00050005,0x00000038,
    0x6f6c5f62,0x5f6c6163,0x00000079,0x00050005,0x0000003e,0x68737570,0x636f6c42,0x0000006b,
    0x00040006,0x0000003e,0x00000000,0x0000706f,0x00030005,0x00000040,0x00000070,0x00040005,
    0x0000004c,0x75706e49,0x00003074,0x00060006,0x0000004c,0x00000000,0x67616d69,0x61645f65,
    0x00006174,0x00030005,0x0000004e,0x00000000,0x00040005,0x00000050,0x7074754f,0x00007475,
    0x00060006,0x00000050,0x00000000,0x4d74756f,0x645f7461,0x00617461,0x00030005,0x00000052,
    0x00000000,0x00040005,0x00000058,0x61687361,0x00006572,0x00040005,0x0000005c,0x61687362,
    0x00006572,0x00040047,0x0000000e,0x0000000b,0x0000001a,0x00040047,0x0000001d,0x0000000b,
    0x0000001b,0x00050048,0x0000003e,0x00000000,0x00000023,0x00000000,0x00030047,0x0000003e,
    0x00000002,0x00040047,0x00000040,0x00000022,0x00000000,0x00040047,0x00000040,0x00000021,
    0x00000002,0x00040047,0x0000004b,0x00000006,0x00000004,0x00040048,0x0000004c,0x00000000,
    0x00000018,0x00050048,0x0000004c,0x00000000,0x00000023,0x00000000,0x00030047,0x0000004c,
    0x00000003,0x00040047,0x0000004e,0x00000022,0x00000000,0x00040047,0x0000004e,0x00000021,
    0x00000000,0x00040047,0x0000004f,0x00000006,0x00000004,0x00040048,0x00000050,0x00000000,
    0x00000019,0x00050048,0x00000050,0x00000000,0x00000023,0x00000000,0x00030047,0x00000050,
    0x00000003,0x00040047,0x00000052,0x00000022,0x00000000,0x00040047,0x00000052,0x00000021,
    0x00000001,0x00040047,0x0000005e,0x0000000b,0x00000019,0x00020013,0x00000002,0x00030021,
    0x00000003,0x00000002,0x00040015,0x00000008,0x00000020,0x00000001,0x00040020,0x00000009,
    0x00000007,0x00000008,0x00040015,0x0000000b,0x00000020,0x00000000,0x00040017,0x0000000c,
    0x0000000b,0x00000003,0x00040020,0x0000000d,0x00000001,0x0000000c,0x0004003b,0x0000000d,
    0x0000000e,0x00000001,0x0004002b,0x0000000b,0x0000000f,0x00000000,0x00040020,0x00000010,
    0x00000001,0x0000000b,0x0004002b,0x00000008,0x00000014,0x00000040,0x0004002b,0x0000000b,
    0x00000017,0x00000001,0x0004003b,0x0000000d,0x0000001d,0x00000001,0x0004002b,0x00000008,
    0x00000021,0x00000010,0x0004002b,0x00000008,0x0000002c,0x00000020,0x0003001e,0x0000003e,
    0x00000008,0x00040020,0x0000003f,0x00000002,0x0000003e,0x0004003b,0x0000003f,0x00000040,
    0x00000002,0x0004002b,0x00000008,0x00000041,0x00000000,0x00040020,0x00000042,0x00000002,
    0x00000008,0x00030016,0x0000004a,0x00000020,0x0003001d,0x0000004b,0x0000004a,0x0003001e,
    0x0000004c,0x0000004b,0x00040020,0x0000004d,0x00000002,0x0000004c,0x0004003b,0x0000004d,
    0x0000004e,0x00000002,0x0003001d,0x0000004f,0x0000004a,0x0003001e,0x00000050,0x0000004f,
    0x00040020,0x00000051,0x00000002,0x00000050,0x0004003b,0x00000051,0x00000052,0x00000002,
    0x0004002b,0x0000000b,0x00000053,0x00000040,0x0004001c,0x00000054,0x0000004a,0x00000053,
    0x0004002b,0x0000000b,0x00000055,0x00000020,0x0004001c,0x00000056,0x00000054,0x00000055,
    0x00040020,0x00000057,0x00000004,0x00000056,0x0004003b,0x00000057,0x00000058,0x00000004,
    0x0004001c,0x00000059,0x0000004a,0x00000055,0x0004001c,0x0000005a,0x00000059,0x00000053,
    0x00040020,0x0000005b,0x00000004,0x0000005a,0x0004003b,0x0000005b,0x0000005c,0x00000004,
    0x0004002b,0x0000000b,0x0000005d,0x00000100,0x0006002c,0x0000000c,0x0000005e,0x0000005d,
    0x00000017,0x00000017,0x0004002b,0x00000008,0x0000005f,0x00000001,0x0004002b,0x00000008,
    0x00000060,0x00000002,0x0004002b,0x00000008,0x00000061,0x00000003,0x0004002b,0x00000008,
    0x00000062,0x00000004,0x0004002b,0x00000008,0x00000063,0x00000005,0x0004002b,0x00000008,
    0x00000064,0x00000006,0x0004002b,0x00000008,0x00000065,0x00000007,0x0004002b,0x00000008,
    0x00000066,0x00000008,0x0004002b,0x00000008,0x00000067,0x00000009,0x0004002b,0x00000008,
    0x00000068,0x0000000a,0x0004002b,0x00000008,0x00000069,0x0000000b,0x0004002b,0x00000008,
    0x0000006a,0x0000000c,0x0004002b,0x00000008,0x0000006b,0x0000000d,0x0004002b,0x00000008,
    0x0000006c,0x0000000e,0x0004002b,0x00000008,0x0000006d,0x0000000f,0x0004002b,0x00000008,
    0x0000006e,0x00000011,0x0004002b,0x00000008,0x0000006f,0x00000012,0x0004002b,0x00000008,
    0x00000070,0x00000013,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,
    0x00000005,0x0004003b,0x00000009,0x0000000a,0x00000007,0x0004003b,0x00000009,0x00000016,
    0x00000007,0x0004003b,0x00000009,0x0000001c,0x00000007,0x0004003b,0x00000009,0x00000023,
    0x00000007,0x0004003b,0x00000009,0x00000028,0x00000007,0x0004003b,0x00000009,0x0000002e,
    0x00000007,0x0004003b,0x00000009,0x00000033,0x00000007,0x0004003b,0x00000009,0x00000038,
    0x00000007,0x00050041,0x00000010,0x00000011,0x0000000e,0x0000000f,0x0004003d,0x0000000b,
    0x00000012,0x00000011,0x0004007c,0x00000008,0x00000013,0x00000012,0x00050084,0x00000008,
    0x00000015,0x00000013,0x00000014,0x0003003e,0x0000000a,0x00000015,0x00050041,0x00000010,
    0x00000018,0x0000000e,0x00000017,0x0004003d,0x0000000b,0x00000019,0x00000018,0x0004007c,
    0x00000008,0x0000001a,0x00000019,0x00050084,0x00000008,0x0000001b,0x0000001a,0x00000014,
    0x0003003e,0x00000016,0x0000001b,0x00050041,0x00000010,0x0000001e,0x0000001d,0x0000000f,
    0x0004003d,0x0000000b,0x0000001f,0x0000001e,0x0004007c,0x00000008,0x00000020,0x0000001f,
    0x0005008b,0x00000008,0x00000022,0x00000020,0x00000021,0x0003003e,0x0000001c,0x00000022,
    0x00050041,0x00000010,0x00000024,0x0000001d,0x0000000f,0x0004003d,0x0000000b,0x00000025,
    0x00000024,0x0004007c,0x00000008,0x00000026,0x00000025,0x00050087,0x00000008,0x00000027,
    0x00000026,0x00000021,0x0003003e,0x00000023,0x00000027,0x00050041,0x00000010,0x00000029,
    0x0000001d,0x0000000f,0x0004003d,0x0000000b,0x0000002a,0x00000029,0x0004007c,0x00000008,
    0x0000002b,0x0000002a,0x0005008b,0x00000008,0x0000002d,0x0000002b,0x0000002c,0x0003003e,
    0x00000028,0x0000002d,0x00050041,0x00000010,0x0000002f,0x0000001d,0x0000000f,0x0004003d,
    0x0000000b,0x00000030,0x0000002f,0x0004007c,0x00000008,0x00000031,0x00000030,0x00050087,
    0x00000008,0x00000032,0x00000031,0x0000002c,0x0003003e,0x0000002e,0x00000032,0x00050041,
    0x00000010,0x00000034,0x0000001d,0x0000000f,0x0004003d,0x0000000b,0x00000035,0x00000034,
    0x0004007c,0x00000008,0x00000036,0x00000035,0x0005008b,0x00000008,0x00000037,0x00000036,
    0x00000014,0x0003003e,0x00000033,0x00000037,0x00050041,0x00000010,0x00000039,0x0000001d,
    0x0000000f,0x0004003d,0x0000000b,0x0000003a,0x00000039,0x0004007c,0x00000008,0x0000003b,
    0x0000003a,0x00050087,0x00000008,0x0000003c,0x0000003b,0x00000014,0x0003003e,0x00000038,
    0x0000003c,0x000100fd,0x00010038,0x00050036,0x00000002,0x00000006,0x00000000,0x00000003,
    0x000200f8,0x00000007,0x000100fd,0x00010038
};

}}} // namespace cv::dnn::vkcom
