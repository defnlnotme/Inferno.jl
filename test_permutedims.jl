using Inferno

# Test permutedims
arr = zeros(6144, 4)
println("Original shape: ", size(arr))
println("After permutedims: ", size(permutedims(arr)))
println("After transpose: ", size(transpose(arr)))
