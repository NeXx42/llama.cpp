#pragma once

#include <cstdint>

namespace nexx{
    struct ModelParams
    {
        wchar_t modelPath[256];

        int32_t contextSize;
    };
}