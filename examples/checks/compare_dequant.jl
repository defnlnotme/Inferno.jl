#!/usr/bin/env julia
# compare_dequant.jl — Verify Julia dequantization matches Python for every type.
#
# Usage:
#   1. python3 examples/gen_dequant_ref.py
#   2. julia --project=. examples/compare_dequant.jl

using LinearAlgebra, Printf, Statistics

function read_ref(path::String)
    open(path, "r") do io
        # First int64 = number of dimensions
        ndims = read(io, Int64)
        # Next ndims int64s = shape
        shape = Tuple([read(io, Int64) for _ in 1:ndims])
        # Rest = float64 data in C-order (row-major)
        n = prod(shape)
        data = Array{Float64}(undef, n)
        read!(io, data)
        # Reshape in C-order: Julia reshape uses Fortran order, so we need to
        # permute. For C-order data, reshape(shape) in Fortran order gives wrong layout.
        # Instead, read as 1D and reshape with C-order by permuting dimensions.
        if length(shape) <= 1
            return reshape(data, shape)
        else
            # C-order data: reshape with reversed dims, then permute
            return permutedims(reshape(data, reverse(shape)), length(shape):-1:1)
        end
    end
end

const REF_DIR = joinpath(@__DIR__, "dequant_ref")
const GGUF_PATH = "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"

const TEST_CASES = [
    ("F32",    "output_norm.weight"),
    ("F16",    "blk.0.ssm_alpha.weight"),
    ("Q8_0",   "blk.0.ssm_out.weight"),
    ("Q4_K",   "blk.0.ffn_gate.weight"),
    ("Q5_K",   "blk.0.attn_qkv.weight"),
    ("Q6_K",   "token_embd.weight"),
    ("IQ4_XS", "blk.8.ffn_gate.weight"),
]

println("Loading Python references...")
refs = Dict{String, Array{Float64}}()
for f in readdir(REF_DIR)
    endswith(f, ".bin") || continue
    refs[replace(f, ".bin" => "")] = read_ref(joinpath(REF_DIR, f))
end

println("Loading GGUF...")
using Inferno
using .Inferno.Loader: extract_tensor
file = Inferno.GGUF.read_gguf(GGUF_PATH)

function compare()
    println("\n── Dequantization comparison ──\n")
    all_pass = true

    for (qtype, tname) in TEST_CASES
        haskey(refs, qtype) || continue
        ref = refs[qtype]

        t = extract_tensor(file, tname)

        # Convert Julia tensor to Float64 with same shape as Python reference
        if ndims(t) == 1 || size(t, 1) == 1 || size(t, 2) == 1
            j = Float64.(vec(collect(t)))
            @assert length(j) == length(ref) "Size mismatch for $qtype"
        else
            # Python dequantize returns (outer, inner), Julia extract_tensor returns (inner, outer)
            # So transpose Julia to match Python
            j = Float64.(collect(t'))
            @assert size(j) == size(ref) "Shape mismatch for $qtype: Julia=$(size(j)) vs Python=$(size(ref))"
        end

        diff = abs.(j .- ref)
        max_err = maximum(diff)
        rel_err = norm(diff) / (norm(ref) + 1e-10)
        cos_sim = dot(j, ref) / (norm(j) * norm(ref) + 1e-10)

        pass = rel_err < 1e-3 && cos_sim > 0.9999
        status = pass ? "  PASS" : "  FAIL"
        all_pass = all_pass && pass

        println("$status  $qtype  max_err=$(Printf.format(Printf.Format("%.2e"), max_err))  " *
                "rel_err=$(Printf.format(Printf.Format("%.2e"), rel_err))  " *
                "cos_sim=$(Printf.format(Printf.Format("%.8f"), cos_sim))  " *
                "$(tname)")
    end

    println()
    println(all_pass ? "All dequantization types PASS" : "SOME TYPES FAILED - see above")
end

compare()
