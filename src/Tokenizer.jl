module Tokenizer

import Base: show
export BPETokenizer, encode, decode, load_tokenizer

"""
Minimal BPE tokenizer that reads the vocabulary from GGUF metadata.
GGUF stores tokens under `tokenizer.ggml.tokens` and merges under
`tokenizer.ggml.merges`.
"""
struct BPETokenizer
    token_to_id::Dict{String, Int}
    id_to_token::Vector{String}
    merges::Vector{Tuple{String, String}}
    bos_id::Int
    eos_id::Int
end

function load_tokenizer(metadata::Dict{String, Any})
    tokens = get(metadata, "tokenizer.ggml.tokens", String[])
    merges_raw = get(metadata, "tokenizer.ggml.merges", String[])
    bos_id = Int(get(metadata, "tokenizer.ggml.bos_token_id", 1))
    eos_id = Int(get(metadata, "tokenizer.ggml.eos_token_id", 2))

    token_to_id = Dict{String, Int}()
    id_to_token = Vector{String}(undef, length(tokens))
    for (i, tok) in enumerate(tokens)
        s = String(tok)
        token_to_id[s] = i  # 1-indexed
        id_to_token[i] = s
    end

    merges = Tuple{String, String}[]
    for m in merges_raw
        parts = split(String(m), ' ', limit=2)
        if length(parts) == 2
            push!(merges, (parts[1], parts[2]))
        end
    end

    BPETokenizer(token_to_id, id_to_token, merges, bos_id, eos_id)
end

function get_byte_map()
    bs = vcat(collect(UInt8('!'):UInt8('~')), collect(UInt8('¡'):UInt8('¬')), collect(UInt8('®'):UInt8('ÿ')))
    cs = Int.(bs)
    n = 0
    for b in 0:255
        if !(UInt8(b) in bs)
            push!(bs, UInt8(b))
            push!(cs, 256 + n)
            n += 1
        end
    end
    byte_to_char = Dict{UInt8, Char}()
    for (b, c) in zip(bs, cs)
        byte_to_char[UInt8(b)] = Char(c)
    end
    return byte_to_char
end

const BYTE_TO_CHAR = get_byte_map()
const CHAR_TO_BYTE = Dict(v => k for (k, v) in BYTE_TO_CHAR)

"""
    encode(tok, text) -> Vector{Int}

Byte-level BPE encoding.
"""
function encode(tok::BPETokenizer, text::String)
    # Map raw UTF-8 bytes to unicode characters using standard BPE mapping
    symbols = String[]
    for b in Vector{UInt8}(text)
        push!(symbols, string(get(BYTE_TO_CHAR, b, Char(b))))
    end
    # Build a priority lookup: merge pair -> priority (lower = higher priority)
    merge_priority = Dict{Tuple{String,String}, Int}()
    for (i, (a, b)) in enumerate(tok.merges)
        merge_priority[(a, b)] = i
    end

    # Iteratively apply merges
    changed = true
    while changed && length(symbols) > 1
        changed = false
        best_idx = 0
        best_pri = typemax(Int)

        for i in 1:(length(symbols) - 1)
            pair = (symbols[i], symbols[i+1])
            pri = get(merge_priority, pair, typemax(Int))
            if pri < best_pri
                best_pri = pri
                best_idx = i
            end
        end

        if best_idx > 0 && best_pri < typemax(Int)
            merged = symbols[best_idx] * symbols[best_idx + 1]
            deleteat!(symbols, best_idx + 1)
            symbols[best_idx] = merged
            changed = true
        end
    end

    # Convert symbols to IDs
    ids = Int[]
    for s in symbols
        id = get(tok.token_to_id, s, 0)
        if id > 0
            push!(ids, id)
        else
            # Unknown token — try byte fallback <0xHH>
            for b in Vector{UInt8}(s)
                fb = string("<0x", uppercase(string(b, base=16, pad=2)), ">")
                fbid = get(tok.token_to_id, fb, 0)
                push!(ids, fbid > 0 ? fbid : 1)  # fallback to BOS/UNK
            end
        end
    end

    return ids
end

"""
    decode(tok, ids) -> String

Decode token IDs back to a UTF-8 string.
"""
function decode(tok::BPETokenizer, ids::Vector{Int})
    bytes_arr = UInt8[]
    for id in ids
        if 1 <= id <= length(tok.id_to_token)
            s = tok.id_to_token[id]
            m = match(r"^<0x([0-9A-Fa-f]{2})>$", s)
            if m !== nothing
                push!(bytes_arr, parse(UInt8, m.captures[1], base=16))
            else
                for c in s
                    b = get(CHAR_TO_BYTE, c, nothing)
                    if b !== nothing
                        push!(bytes_arr, b)
                    else
                        # Unmapped char (e.g. special unicode tokens) — encode as raw UTF-8
                        for byte in Vector{UInt8}(string(c))
                            push!(bytes_arr, byte)
                        end
                    end
                end
            end
        end
    end
    return String(bytes_arr)
end

function show(io::IO, tok::BPETokenizer)
    print(io, "[CPU] BPETokenizer (vocab=$(length(tok.id_to_token)))")
end

end # module
