#include <llama.h>
#include <string>
#include <vector>
#include <iostream>



extern "C" {

llama_model* llamaModel;
llama_context* llamaContext;
llama_sampler* llamaSampler;

typedef void (*TokenCallback)(const char* message);

__declspec(dllexport) int CreateModel(char* model_path){
    int ngl = 99;
    int n_contextsize = 2048;

    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llamaModel = llama_load_model_from_file(model_path, model_params);
    if (!llamaModel) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return -1;
    }

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_contextsize;
    ctx_params.n_batch = n_contextsize;

    llamaContext = llama_new_context_with_model(llamaModel, ctx_params);
    if (!llamaContext) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return -1;
    }

    // initialize the sampler
    llamaSampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(llamaSampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(llamaSampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(llamaSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    return 0;
}

__declspec(dllexport) int Run(char* prompt, TokenCallback callback){
    // tokenize the prompt
    const int n_prompt_tokens = -llama_tokenize(llamaModel, prompt, strlen(prompt), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);

    if (llama_tokenize(llamaModel, prompt, strlen(prompt), prompt_tokens.data(), prompt_tokens.size(), llama_get_kv_cache_used_cells(llamaContext) == 0, true) < 0) {
        printf("token fail");
        GGML_ABORT("failed to tokenize the prompt\n");
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;

    while (true) {
        // check if we have enough space in the context to evaluate this batch
        int n_ctx = llama_n_ctx(llamaContext);
        int n_ctx_used = llama_get_kv_cache_used_cells(llamaContext);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            fprintf(stderr, "context size exceeded\n");
            exit(0);
        }

        if (llama_decode(llamaContext, batch)) {
            GGML_ABORT("failed to decode\n");
        }

        // sample the next token
        new_token_id = llama_sampler_sample(llamaSampler, llamaContext, -1);

        // is it an end of generation?
        if (llama_token_is_eog(llamaModel, new_token_id)) {
            break;
        }

        // convert the token to a string, print it and add it to the response
        char buf[256];
        int n = llama_token_to_piece(llamaModel, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            GGML_ABORT("failed to convert token to piece\n");
        }

        std::string piece(buf, n);
        callback(piece.c_str());

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    return 0;
};

__declspec(dllexport) int FreeModel(){
    llama_sampler_free(llamaSampler);
    llama_free(llamaContext);
    llama_free_model(llamaModel);

    return 0;
}

}