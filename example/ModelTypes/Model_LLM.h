#include <IModel.h>
#include <llama.h>
#include <ModelParams.h>

namespace nexx{

    class Model_LLM : public IModel{

    private:
        llama_model* llamaModel;
        llama_context* llamaContext;
        llama_sampler* llamaSampler;

    public:
        int SetupModel(ModelParams params) override;
        void Run(std::string prompt, TokenGeneratedCallback onTokenGenerate) override;
    };

}