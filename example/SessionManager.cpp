#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <wtypes.h>
#include <thread>
#include <map>

#include <llama.h>
#include <Session.h>
#include <SessionManager.h>

#include <IModel.h>
#include <ModelTypes.h>

#include <Model_LLM.h>

namespace nexx{
    void SessionManager::AddClient(ModelTypes model, HANDLE hPipe){
        Session* newClientSession = new Session();
        newClientSession->HandleClient(hPipe, this->activeModels[model]);

        activeClients.push_back(newClientSession);
    }

    void SessionManager::MonitorPipeline(ModelTypes model){

        while (true) {
            std::wstring pipelineName = modelsLookup.at(model);

            HANDLE hPipe = CreateNamedPipeW(
                (L"\\\\.\\pipe\\" + pipelineName).c_str(),
                PIPE_ACCESS_DUPLEX,             // Read/Write access
                PIPE_TYPE_BYTE | PIPE_WAIT,     // Byte-type pipe, blocking mode
                PIPE_UNLIMITED_INSTANCES,       // Unlimited instances
                512,                            // Output buffer size
                512,                            // Input buffer size
                0,                              // Default timeout
                nullptr                         // Default security attributes
            );

            if (hPipe == INVALID_HANDLE_VALUE) {
                std::cerr << "Failed to create named pipe. Error: " << GetLastError() << std::endl;
                break;
            }

            // Wait for a client to connect
            std::wcout << L"Waiting for a client to connect..." << std::endl;
            BOOL connected = ConnectNamedPipe(hPipe, nullptr) || GetLastError() == ERROR_PIPE_CONNECTED;

            if (connected) {
                this->AddClient(model, hPipe);
            } else {
                CloseHandle(hPipe);
            }
        }

    }

    int SessionManager::Setup(){
        activeModels.insert({ ModelTypes::LLM, new Model_LLM()});

        for (auto& pair : activeModels){
            pair.second->SetupModel(ModelParams{
                L"E:/Projects/AI/LLM/LLMProcessor/LLMProcessor/Models/Llama-3.2-1B-Instruct.fp16.gguf",
                4096
            });

            // this needs to be a detached thread
            this->MonitorPipeline(pair.first);
        }

        return 0;
    }
}