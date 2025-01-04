#include <string>
#include <functional>

#include <ModelParams.h>

#pragma once

namespace nexx{
    class IModel{
    public:
        using TokenGeneratedCallback = std::function<void(const std::string)>;

        virtual int SetupModel(ModelParams params) = 0;
        virtual void Run(std::string prompt, TokenGeneratedCallback onTokenGenerate) = 0;
    };
}