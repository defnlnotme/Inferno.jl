push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Inferno
using Inferno.QuantsData
using Inferno.Dequant
using Test

println("Verifying dequantization algorithms...")

@testset "IQ2_XXS Dequantization" begin
    # block_iq2_xxs: 2 bytes d, 64 bytes qs = 66 bytes total for 256 elements
    data = zeros(UInt8, 66)
    # Set scale d = 1.0 (u16 in f16 format)
    # 1.0f0 in Float16 is 0x3c00
    data[1] = 0x00
    data[2] = 0x3c
    
    # All quants = 0, should result in all 0s
    y = dequantize_iq2_xxs(data, 256)
    @test length(y) == 256
    @test all(y .== 0.0f0)
    println("IQ2_XXS zero test passed.")
    
    # Try non-zero
    data[3] = 0x01 # first uint16 in qs = 1
    # grid[1] is 0x080808080808082b
    # db calculation in code: db = d * (0.5 + (aux32_2 >> 28)) * 0.25
    # aux32_2 is the last 4 bytes of the 8-byte block.
    # The first 32 elements use data[3:10]. aux32_1 = data[3:6], aux32_2 = data[7:10].
    # if data[3]=1, aux32_1 = 1.
    # aux32_2 >> 28 will be 0. db = 1.0 * 0.5 * 0.25 = 0.125
    # grid index for l=0: (aux32_1 >> 0) & 0xFF = 1. Julia grid index = 2.
    # grid_val = IQ2XXS_GRID[2] = 0x080808080808082b
    # signs_idx = (aux32_2 >> 0) & 127 = 0. Julia signs index = 1.
    # signs = KSIGNS_IQ2XS[1] = 0
    # y[1..8] = db * byte_val * 1.0
    # byte_val[1] = 0x2b = 43. y[1] = 0.125 * 43 = 5.375
    # byte_val[2..8] = 0x08 = 8. y[2..8] = 0.125 * 8 = 1.0
    y = dequantize_iq2_xxs(data, 256)
    @test y[1] == 5.375f0
    @test all(y[2:8] .== 1.0f0)
    println("IQ2_XXS value test passed.")
end

println("All algorithms verified successfully.")
