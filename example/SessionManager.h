#include <wtypes.h>
#include <map>

#include <Session.h>

#include <IModel.h>
#include <ModelTypes.h>

namespace nexx{
    const std::map<ModelTypes, wchar_t*> modelsLookup = {
        {ModelTypes::LLM, L"LLM"}
    };


    class SessionManager
    {
    private:
        std::map<ModelTypes, IModel*> activeModels;


        std::vector<Session*> activeClients;

        void CreatePipeline();

    public:
        int Setup();

        void AddClient(ModelTypes model, HANDLE hPipe);
        void MonitorPipeline(ModelTypes model);
    };
}