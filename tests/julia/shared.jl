# shared.jl — Common utilities for Julia tests
using LinearAlgebra, Statistics

function read_npy(path::String)
    open(path, "r") do io
        magic = read(io, 6)
        major = read(io, UInt8); read(io, UInt8)
        hl = major == 1 ? Int(read(io, UInt16)) : Int(read(io, UInt32))
        hs = String(read(io, hl))
        dtype_str = match(r"'descr':\s*'([^']+)'", hs).captures[1]
        shape_str = match(r"'shape':\s*\(([^)]*)\)", hs).captures[1]
        parts = filter(!isempty, strip.(split(shape_str, ',')))
        shape = isempty(parts) ? () : Tuple(parse.(Int, parts))
        tc = dtype_str[2]; ts = parse(Int, dtype_str[3:end])
        T = tc == 'f' ? (ts == 4 ? Float32 : Float64) : error("bad dtype")
        shape == () && return read(io, T)
        data = Array{T}(undef, shape...)
        read!(io, data)
        return data
    end
end

function assert_close(julia_vec, ref_vec, name; rtol=1e-3)
    j = Float64.(vec(julia_vec))
    r = Float64.(vec(ref_vec))
    @assert length(j) == length(r) "Size mismatch for $name"
    rel = norm(j .- r) / (norm(r) + 1e-10)
    cos_sim = dot(j, r) / (norm(j) * norm(r) + 1e-10)
    pass = rel < rtol && cos_sim > 0.9999
    status = pass ? "PASS" : "FAIL"
    println("  $status  $name  rel_err=$(round(rel, sigdigits=4))  cos_sim=$(round(cos_sim, digits=8))")
    @assert pass "$name failed: rel_err=$rel, cos_sim=$cos_sim"
end

const TMP_DIR = "/tmp"
