#!/usr/bin/env bash

cd /var/home/fra/dev/inferno && julia --project=. -e '
using Inferno
using Inferno.Model
using Inferno.Tokenizer

model_path = "/var/home/fra/dev/inferno/tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
model, tok = load_model(model_path)
caches = [init_kv_cache(model.config) for _ in 1:model.config.num_hidden_layers]

prompt = "The capital of France is"
tokens = Tokenizer.encode(tok, prompt)
println("Tokens: ", tokens)

logits = Model.forward!(model, tokens, 0, caches)
last_logits = vec(logits[:, end])

for i in 1:3
 tok_id = argmax(last_logits)
 tok_str = Tokenizer.decode(tok, [tok_id])
 println("Token $i: ", tok_id, " -> \"", tok_str, "\"")

 curr_pos = caches[1].pos
 next_logits = Model.forward!(model, [tok_id], curr_pos, caches)
 global last_logits = vec(next_logits[:, 1])
end
'
