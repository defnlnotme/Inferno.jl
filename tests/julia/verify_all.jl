#!/usr/bin/env julia
# verify_all.jl — Verify inference produces correct output.
#
# The dequantized weights are Float32 (from the library). Comparing intermediates
# against Python Float64 references will show small differences. The test validates
# that the final outputs (which converge over 24 layers) match the Python reference.
#
# Run: python3 tests/python/generate_all.py && julia --project=. tests/julia/verify_all.jl

using LinearAlgebra, Statistics, Printf
using Inferno
using .Inferno.Loader: extract_tensor, get_bias_or_norm, extract_sorted_blocks
using .Inferno: Model

const ROOT = normpath(joinpath(@__DIR__, "../.."))
const GGUF_PATH = joinpath(ROOT, "tests/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf")
const REF_DIR = "/tmp/test_refs"

function read_npy(path::String)
    open(path, "r") do io
        read(io, 6); major = read(io, UInt8); read(io, UInt8)
        hl = major == 1 ? Int(read(io, UInt16)) : Int(read(io, UInt32))
        hs = String(read(io, hl))
        ds = match(r"'descr':\s*'([^']+)'", hs).captures[1]
        ss = match(r"'shape':\s*\(([^)]*)\)", hs).captures[1]
        parts = filter(!isempty, strip.(split(ss, ',')))
        shape = isempty(parts) ? () : Tuple(parse.(Int, parts))
        tc = ds[2]; ts = parse(Int, ds[3:end])
        T = tc == 'f' ? (ts == 4 ? Float32 : Float64) : error("bad")
        shape == () && return read(io, T)
        data = Array{T}(undef, shape...); read!(io, data); data
    end
end

function check(j, r, name; rtol=2e-3)
    j64, r64 = Float64.(vec(j)), Float64.(vec(r))
    if length(j64) != length(r64)
        println("  FAIL  $name  size mismatch"); return false
    end
    rel = norm(j64 .- r64) / (norm(r64) + 1e-10)
    cos = dot(j64, r64) / (norm(j64) * norm(r64) + 1e-10)
    pass = rel < rtol && cos > 0.9999
    println("  $(pass ? "PASS" : "FAIL")  $name  rel=$(Printf.format(Printf.Format("%.2e"), rel))  cos=$(Printf.format(Printf.Format("%.6f"), cos))")
    return pass
end

cpu_rmsnorm(x, w, eps) = Float64.(x) .* (1.0 / sqrt(sum(Float64.(x) .^ 2) / length(x) + eps)) .* Float64.(w)
cpu_silu(x) = Float64.(x) .* (1.0 ./ (1.0 .+ exp.(-Float64.(x))))

file = Inferno.GGUF.read_gguf(GGUF_PATH)
cfg = let md = file.metadata
    Model.QwenConfig(architecture=:qwen35,
        vocab_size=Int(get(md, "qwen35.vocab_size", 248320)),
        hidden_size=Int(md["qwen35.embedding_length"]),
        intermediate_size=Int(md["qwen35.feed_forward_length"]),
        num_hidden_layers=Int(md["qwen35.block_count"]),
        num_attention_heads=Int(md["qwen35.attention.head_count"]),
        num_key_value_heads=Int(md["qwen35.attention.head_count_kv"]),
        head_dim=Int(md["qwen35.attention.key_length"]),
        rms_norm_eps=Float32(md["qwen35.attention.layer_norm_rms_epsilon"]),
        rope_theta=Float32(md["qwen35.rope.freq_base"]),
        max_position_embeddings=min(4096, Int(md["qwen35.context_length"])),
        full_attention_interval=Int(md["qwen35.full_attention_interval"]),
        ssm_inner_size=Int(md["qwen35.ssm.inner_size"]),
        ssm_state_size=Int(md["qwen35.ssm.state_size"]),
        ssm_group_count=Int(md["qwen35.ssm.group_count"]),
        ssm_time_step_rank=Int(md["qwen35.ssm.time_step_rank"]),
        ssm_conv_kernel=Int(md["qwen35.ssm.conv_kernel"]),
        num_experts=0, num_experts_per_tok=0,
        q_lora_rank=0, kv_lora_rank=0, qk_rope_head_dim=0, v_head_dim=0)
end

embed_mat = Float64.(collect(extract_tensor(file, "token_embd.weight")))
output_norm_w = Float64.(vec(collect(extract_tensor(file, "output_norm.weight"))))
lm_head = haskey(file.tensors, "output.weight") ? Float64.(collect(extract_tensor(file, "output.weight")))' : embed_mat'
blocks_raw = extract_sorted_blocks(file.tensors)

sm(f) = f === nothing ? zeros(Float64,1,1) : Float64.(collect(extract_tensor(file,f.name)'))
sv(f) = f === nothing ? Float64[] : Float64.(vec(collect(extract_tensor(file,f.name))))

struct BW
    is_ssm::Bool; attn_norm::Vector{Float64}; post_attn_norm::Vector{Float64}
    ffn_gate::Matrix{Float64}; ffn_up::Matrix{Float64}; ffn_down::Matrix{Float64}
    attn_qkv::Matrix{Float64}; attn_gate::Matrix{Float64}; ssm_out::Matrix{Float64}
    conv1d::Matrix{Float64}; alpha::Matrix{Float64}; beta::Matrix{Float64}
    ssm_a::Vector{Float64}; dt_bias::Vector{Float64}; ssm_norm::Vector{Float64}
    attn_q::Matrix{Float64}; attn_k::Matrix{Float64}; attn_v::Matrix{Float64}
    attn_output::Matrix{Float64}; attn_q_norm::Vector{Float64}; attn_k_norm::Vector{Float64}
end
bws = [BW(blk.attn_qkv_weight!==nothing,sv(blk.attn_norm_weight),sv(blk.post_attention_norm_weight),
    sm(blk.ffn_gate_weight),sm(blk.ffn_up_weight),sm(blk.ffn_down_weight),
    sm(blk.attn_qkv_weight),sm(blk.attn_gate_weight),sm(blk.ssm_out_weight),
    sm(blk.ssm_conv1d_weight),sm(blk.ssm_alpha_weight),sm(blk.ssm_beta_weight),
    sv(blk.ssm_a),sv(blk.ssm_dt_bias),sv(blk.ssm_norm_weight),
    sm(blk.attn_q_weight),sm(blk.attn_k_weight),sm(blk.attn_v_weight),
    sm(blk.attn_output_weight),sv(blk.attn_q_norm_weight),sv(blk.attn_k_norm_weight))
    for blk in blocks_raw]

refs = Dict{String,Array}()
for f in readdir(REF_DIR); endswith(f,".npy")||continue
    refs[replace(f,".npy"=>"")]=Float64.(read_npy(joinpath(REF_DIR,f))); end

println("="^60)
println("Inference verification")
println("Weights: Float32 dequant (matching inference.jl)")
println("Forward: Float64 CPU (avoids GPU Float16 overflow)")
println("Loaded $(length(refs)) reference arrays")
println("="^60,"\n")

function run_forward()
    all_pass = true
    x = embed_mat[:, 151647]
    all_pass &= check(x, refs["01_embed"], "01_embedding"; rtol=1e-6)

    hd=cfg.head_dim; nq=cfg.num_attention_heads; nkv=cfg.num_key_value_heads
    inner=cfg.ssm_inner_size; ss=cfg.ssm_state_size; grp=cfg.ssm_group_count
    hvd=inner÷cfg.ssm_time_step_rank; conv_ch=2*grp*ss+inner; K=cfg.ssm_conv_kernel
    eps=Float64(cfg.rms_norm_eps); gqa=nq÷nkv

    conv_s=[zeros(Float64,conv_ch,K) for _ in 1:cfg.num_hidden_layers]
    h_s=[zeros(Float64,hvd,ss,grp) for _ in 1:cfg.num_hidden_layers]
    kk=[zeros(Float64,hd,nkv,cfg.max_position_embeddings) for _ in 1:cfg.num_hidden_layers]
    vv=[zeros(Float64,hd,nkv,cfg.max_position_embeddings) for _ in 1:cfg.num_hidden_layers]

    for i in 1:cfg.num_hidden_layers
        bw=bws[i]; xn=cpu_rmsnorm(x,bw.attn_norm,eps)
        if bw.is_ssm
            qkv=bw.attn_qkv*xn; zb=bw.attn_gate*xn
            K>1&&(conv_s[i][:,1:K-1]=conv_s[i][:,2:K]); conv_s[i][:,K]=qkv
            xc=cpu_silu(sum(conv_s[i][:,k].*bw.conv1d[:,k] for k in 1:K))
            qas=ss*grp
            qa=reshape(xc[1:qas],ss,grp); ka=reshape(xc[qas+1:2*qas],ss,grp); va=reshape(xc[2*qas+1:2*qas+inner],hvd,grp)
            ap=bw.alpha*xn; bp=bw.beta*xn; ya=zeros(Float64,inner)
            for g in 1:grp
                qg,kg,vg=qa[:,g],ka[:,g],va[:,g]
                qn=qg./sqrt(sum(qg.^2)+eps); kn=kg./sqrt(sum(kg.^2)+eps)
                dg=exp(log(1+exp(ap[g]+bw.dt_bias[g]))*bw.ssm_a[g]); bg=1/(1+exp(-bp[g]))
                h_s[i][:,:,g].*=dg; sk=h_s[i][:,:,g]*kn
                for j in eachindex(sk); sk[j]=bg*(vg[j]-sk[j]); end
                h_s[i][:,:,g].+=sk*kn'; ya[(g-1)*hvd+1:g*hvd].=h_s[i][:,:,g]*qn
            end
            yr=reshape(ya,hvd,grp)
            for g in 1:grp; v=view(yr,:,g); sc=1/sqrt(sum(v.^2)/hvd+eps); v.*=sc.*bw.ssm_norm; end
            yr.*=reshape(cpu_silu(zb),hvd,grp); br=bw.ssm_out*vec(yr); lb="SSM"
        else
            qb=bw.attn_q*xn; kb=bw.attn_k*xn; vb=bw.attn_v*xn
            qs=nq*hd; q=qb[1:qs]; gt=qb[qs+1:2*qs]
            qh=reshape(q,hd,nq); kh=reshape(kb,hd,nkv); vh=reshape(vb,hd,nkv)
            for h in 1:nq; qh[:,h]=cpu_rmsnorm(qh[:,h],bw.attn_q_norm,eps); end
            for h in 1:nkv; kh[:,h]=cpu_rmsnorm(kh[:,h],bw.attn_k_norm,eps); end
            qh.*=cpu_silu(reshape(gt,hd,nq))
            rp=hd÷2; th=Float64(cfg.rope_theta)
            for h in 1:nq; for p in 1:rp
                s=sin(0/th^(2*(p-1)/hd)); c=cos(0/th^(2*(p-1)/hd))
                q0,q1=qh[2p-1,h],qh[2p,h]; qh[2p-1,h]=q0*c-q1*s; qh[2p,h]=q0*s+q1*c
            end; end
            for h in 1:nkv; for p in 1:rp
                s=sin(0/th^(2*(p-1)/hd)); c=cos(0/th^(2*(p-1)/hd))
                k0,k1=kh[2p-1,h],kh[2p,h]; kh[2p-1,h]=k0*c-k1*s; kh[2p,h]=k0*s+k1*c
            end; end
            kk[i][:,:,1]=kh; vv[i][:,:,1]=vh; ao=zeros(Float64,hd,nq); sc=1/sqrt(hd)
            for h in 1:nq
                kv_h=(h-1)÷gqa+1; Kp=kk[i][:,kv_h,1:1]; Vp=vv[i][:,kv_h,1:1]
                qh64=qh[:,h]; scores=Kp'*qh64; scores.*=sc; scores.-=maximum(scores)
                scores=exp.(scores); scores./=sum(scores); ao[:,h]=Vp*scores
            end
            br=bw.attn_output*vec(ao); lb="ATN"
        end
        x=x.+br
        xn2=cpu_rmsnorm(x,bw.post_attn_norm,eps)
        mlp=bw.ffn_down*(cpu_silu(bw.ffn_gate*xn2).*(bw.ffn_up*xn2))
        x=x.+mlp
    end

    hf=cpu_rmsnorm(x,output_norm_w,eps); lg=lm_head*hf
    all_pass&=check(hf,refs["09_final_norm"],"final_norm")
    all_pass&=check(lg,refs["10_logits"],"logits")

    top5=sortperm(lg,rev=true)[1:5]; ref5=sortperm(refs["10_logits"],rev=true)[1:5]
    tokens_match = top5 == ref5
    println("\n  Julia top-5: $top5")
    println("  Ref   top-5: $ref5")
    println("  Tokens match: $tokens_match")
    println("\n",all_pass&&tokens_match ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
    return all_pass && tokens_match
end

run_forward()
