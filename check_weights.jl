using Inferno
using LinearAlgebra

function check_weights()
    model, tokenizer = Inferno.load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
    
    # Check embedding
    println("Embedding shape: ", size(model.embed))
    println("Embedding sample values: ", model.embed[1:5, 761])
    
    # Check lm_head
    println("\nLM head shape: ", size(model.lm_head))
    println("LM head sample values (first 5): ", model.lm_head[1:5, 1:5])
    
    # Check if lm_head is tied to embedding
    println("\nAre lm_head and embed' the same? ", model.lm_head == model.embed')
    
    # Check final norm
    println("\nFinal norm weight shape: ", size(model.final_norm.weight))
    println("Final norm weight sample: ", model.final_norm.weight[1:5])
end

check_weights()
