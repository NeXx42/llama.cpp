#include <Session.h>
#include <wtypes.h>
#include <iostream>
#include <thread>

#include <llama.h>
#include <string>
#include <ggml-backend.h>
#include <xlocbuf>
#include <codecvt>
#include <vector>
#include "IModel.h"


namespace nexx{
    void Session::SendBackToClient(const std::string toSend){
        std::cout << "Sending: " << toSend << std::endl;
        WriteFile(this->hPipe, toSend.c_str(), toSend.size(), nullptr, nullptr);
        FlushFileBuffers(this->hPipe);
    }

    void Session::InternalHandle(IModel* model){
        std::wcout << L"Client connected on thread: " << std::this_thread::get_id() << std::endl;

        BOOL result = false;
        DWORD bytesRead;
        DWORD bytesWritten;

        IModel::TokenGeneratedCallback callback = std::bind(&Session::SendBackToClient, this, std::placeholders::_1);

        char buffer[2048];
        
        const char endMarker = 0xFF; // The end marker as a byte
        std::vector<char> chunkedData; // Use vector for accumulating bytes


        while (true)
        {
            result = ReadFile(hPipe, buffer, sizeof(buffer), &bytesRead, nullptr);

            if (result && bytesRead > 0) {
                chunkedData.insert(chunkedData.end(), buffer, buffer + bytesRead);
                auto it = std::find(chunkedData.begin(), chunkedData.end(), endMarker);
    
                if (it != chunkedData.end())
                {
                    std::string res = std::string(chunkedData.begin(), it);
                    chunkedData.clear();

                    model->Run(res, callback);
                }
                
            }
            else if (!result) {
                // Handle ReadFile failure (optional, based on your requirements)
                break;
            }
        }
        
        printf("Completed");
        std::cin.get();

        CloseHandle(this->hPipe);
    }

    void Session::HandleClient(HANDLE hPipe, IModel* model){
        this->hPipe = hPipe;

        std::thread clientThread([this, model]() {
            this->InternalHandle(model);
        });

        clientThread.detach();
    }
}