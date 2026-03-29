import json
with open("tests/models/Qwen3.5-0.8B/config.json") as f:
    c = json.load(f)
    tc = c['text_config']
    print(f"vocab_size: {tc.get('vocab_size', 'not found')}")
    print(f"full_attention_interval: {tc.get('full_attention_interval')}")
    print(f"layer_types: {tc.get('layer_types', [])}")
