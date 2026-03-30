// Dump intermediate tensor values from llama.cpp for debugging
// Build with: g++ -O2 -I.. -I../include dump_tensors.cpp -L../build/ggml/src -lggml -L../build/src -lllama -o dump_tensors
// Or use cmake

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>

// Structure to track which tensors we want to dump
struct DumpConfig {
    std::vector<std::string> target_names;  // Tensor name patterns to dump
    int max_values;                          // Max values to print per tensor
    bool print_norm;                         // Print norm instead of all values
    FILE* output_file;
};

// Global config
DumpConfig g_dump_config;

// Compute L2 norm of a tensor
float compute_norm(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i] * data[i];
    }
    return std::sqrt(sum);
}

// Check if tensor name matches any target pattern
bool should_dump(const char* name, const DumpConfig& config) {
    for (const auto& pattern : config.target_names) {
        if (strstr(name, pattern.c_str()) != nullptr) {
            return true;
        }
    }
    return false;
}

// Callback function for tensor evaluation
bool tensor_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    DumpConfig* config = (DumpConfig*)user_data;
    
    // During ask phase, indicate if we want to observe this tensor
    if (ask) {
        return should_dump(t->name, *config);
    }
    
    // During evaluation phase, dump the tensor data
    const char* name = t->name;
    if (!should_dump(name, *config)) {
        return true;  // Continue execution
    }
    
    // Get tensor data
    float* data = (float*)t->data;
    size_t nelements = ggml_nelements(t);
    
    fprintf(config->output_file, "TENSOR: %s, shape: [");
    for (int i = 0; i < t->n_dims; ++i) {
        fprintf(config->output_file, "%ld%s", (long)t->ne[i], i < t->n_dims - 1 ? ", " : "");
    }
    fprintf(config->output_file, "]\n");
    
    if (config->print_norm) {
        float norm = compute_norm(data, nelements);
        fprintf(config->output_file, "  norm: %.6f, elements: %zu\n", norm, nelements);
    } else {
        fprintf(config->output_file, "  values: ");
        int n_print = std::min(config->max_values, (int)nelements);
        for (int i = 0; i < n_print; ++i) {
            fprintf(config->output_file, "%.6f%s", data[i], i < n_print - 1 ? ", " : "");
        }
        if ((int)nelements > n_print) {
            fprintf(config->output_file, " ... (%zu more)", nelements - n_print);
        }
        fprintf(config->output_file, "\n");
    }
    
    return true;  // Continue execution
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_path> <prompt> [output_file]\n", argv[0]);
        fprintf(stderr, "Dumps intermediate tensor values for debugging.\n");
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* prompt = argv[2];
    const char* output_path = argc > 3 ? argv[3] : "tensor_dump.txt";
    
    // Configure which tensors to dump
    g_dump_config.target_names = {
        "output",      // Final output tensor
        "norm",        // Normalization layers
        "residual",    // Residual additions
        "result",      // Attention/SSM results
        "l_out",       // Layer outputs (llama naming)
        // Add more patterns as needed
    };
    g_dump_config.max_values = 10;
    g_dump_config.print_norm = true;  // Print norms for easier comparison
    g_dump_config.output_file = fopen(output_path, "w");
    if (!g_dump_config.output_file) {
        fprintf(stderr, "Failed to open output file: %s\n", output_path);
        return 1;
    }
    
    // Initialize llama
    llama_backend_init();
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model from: %s\n", model_path);
        return 1;
    }
    
    // Create context with callback
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.cb_eval = tensor_callback;
    ctx_params.cb_eval_user_data = &g_dump_config;
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }
    
    // Tokenize prompt
    std::vector<llama_token> tokens;
    int n_tokens = -llama_tokenize(model, prompt, strlen(prompt), nullptr, 0, true, true);
    tokens.resize(n_tokens);
    llama_tokenize(model, prompt, strlen(prompt), tokens.data(), n_tokens, true, true);
    
    fprintf(g_dump_config.output_file, "Prompt: \"%s\"\n", prompt);
    fprintf(g_dump_config.output_file, "Tokens: ");
    for (auto t : tokens) {
        fprintf(g_dump_config.output_file, "%d ", t);
    }
    fprintf(g_dump_config.output_file, "\n\n");
    
    // Process prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_decode(ctx, batch);
    
    fprintf(g_dump_config.output_file, "\n=== Token processing complete ===\n");
    
    // Get logits for the last position
    float* logits = llama_get_logits(ctx);
    int vocab_size = llama_n_vocab(model);
    
    // Find top 10 tokens
    std::vector<std::pair<float, int>> logit_pairs;
    for (int i = 0; i < vocab_size; ++i) {
        logit_pairs.push_back({logits[i], i});
    }
    std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + 10, logit_pairs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    fprintf(g_dump_config.output_file, "\nTop 10 predictions:\n");
    for (int i = 0; i < 10; ++i) {
        const char* tok_str = llama_token_get_text(model, logit_pairs[i].second);
        fprintf(g_dump_config.output_file, "  %d: '%s' (logit: %.4f)\n", 
                logit_pairs[i].second, tok_str, logit_pairs[i].first);
    }
    
    // Cleanup
    fclose(g_dump_config.output_file);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    printf("Tensor dump written to: %s\n", output_path);
    return 0;
}
