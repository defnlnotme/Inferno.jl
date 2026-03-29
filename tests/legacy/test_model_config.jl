using Inferno
using Printf

println("Loading model...")
model, _ = load_model_cpu("tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")

# Check model config
println("\nModel config:")
println(" hidden_size: ", model.config.hidden_size)
println(" intermediate_size: ", model.config.intermediate_size)
println(" num_attention_heads: ", model.config.num_attention_heads)
println(" num_key_value_heads: ", model.config.num_key_value_heads)
println(" num_hidden_layers: ", model.config.num_hidden_layers)
println(" vocab_size: ", model.config.vocab_size)
println(" rms_norm_eps: ", model.config.rms_norm_eps)
println(" rope_theta: ", model.config.rope_theta)

# Check number of parameters
function count_params(model)
 total = 0
 # Embedding
 total += length(model.embed)
 # LM head (same as embedding, tied)
 # Layers
 for layer in model.layers
 if hasfield(typeof(layer), 'op') && layer.is_ssm
 total += length(layer.op.in_proj.weight)
 total += length(layer.op.out_proj.weight)
 total += length(layer.op.conv1d.weight)
 total += length(layer.op.conv1d.bias)
 total += length(layer.op.A_log)
 total += length(layer.op.D)
 elseif layer.is_attention
 # Attention weights
 total += length(layer.op.q_proj.weight)
 total += length(layer.op.k_proj.weight)
 total += length(layer.op.v_proj.weight)
 total += length(layer.op.o_proj.weight)
 end
 # MLP
 total += length(layer.mlp.gate_weight)
 total += length(layer.mlp.up_weight)
 total += length(layer.mlp.down_weight)
 # Norms
 total += length(layer.input_norm.weight)
 total += length(layer.post_norm.weight)
 end
 # Final norm
 total += length(model.final_norm.weight)
 return total
end

println("\nApproximate parameter count: ", count_params(model))
