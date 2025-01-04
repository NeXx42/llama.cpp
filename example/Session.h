#pragma once

#include <wtypes.h>
#include <string>

#include <llama.h>
#include <IModel.h>

namespace nexx
{
    class Session
    {
    private:
        llama_model* llamaModel;
        llama_context* llamaContext;
        llama_sampler* llamaSampler;

        HANDLE hPipe;
        
        void InternalHandle(IModel* model);
        void SendBackToClient(std::string toSend);
    public:
        void HandleClient(HANDLE hPipe, IModel* model);
    };
}