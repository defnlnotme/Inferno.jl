using Inferno
using LinearAlgebra

function test_reshape()
    # Create a simple test case
    # GGUF tensor [2, 3] with values [[1, 2, 3], [4, 5, 6]]
    # Stored as: 1, 2, 3, 4, 5, 6 (row-major)
    data = Float32[1, 2, 3, 4, 5, 6]
    
    # dims = (2, 3), inner = 2, outer = 3
    inner = 2
    outer = 3
    
    println("=== Test Reshape ===")
    println("Data (GGUF row-major): ", data)
    println("GGUF shape: [2, 3] = 2 rows, 3 columns")
    println("Expected matrix:")
    println("  [1 2 3]")
    println("  [4 5 6]")
    
    # Old (wrong) reshape
    wrong = reshape(data, inner, outer)
    println("\nOld reshape(data, inner, outer):")
    println("  Shape: ", size(wrong))
    println("  Result:")
    for i in 1:size(wrong, 1)
        println("  ", wrong[i, :])
    end
    
    # Correct reshape
    correct = reshape(data, outer, inner)'
    println("\nNew reshape(data, outer, inner)':")
    println("  Shape: ", size(correct))
    println("  Result:")
    for i in 1:size(correct, 1)
        println("  ", correct[i, :])
    end
    
    # Check if correct matches expected
    expected = Float32[1 2 3; 4 5 6]
    println("\nExpected:")
    for i in 1:size(expected, 1)
        println("  ", expected[i, :])
    end
    
    println("\nMatch: ", correct == expected)
end

test_reshape()
