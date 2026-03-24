#!/bin/bash
# Inferno Test Runner
# Usage:
#   ./run.sh              # Run all tests
#   ./run.sh unit         # Run only unit tests (fast, no model required)
#   ./run.sh model        # Run only model-dependent tests
#   ./run.sh integration  # Run only integration tests
#   ./run.sh coverage     # Run with coverage tracking

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_status() {
    echo -e "${YELLOW}[TEST] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[PASS] $1${NC}"
}

print_error() {
    echo -e "${RED}[FAIL] $1${NC}"
}

# Default: run all tests
TEST_MODE="${1:-all}"

case "$TEST_MODE" in
    unit)
        print_header "Running Unit Tests (no model required)"
        print_status "These tests run in isolation without GPU/model dependencies"
        
        julia --project=. -e '
            using Test
            println("Testing Engine module...")
            include("unit/test_engine.jl")
            println("\nTesting Tokenizer module...")
            include("unit/test_tokenizer.jl")
            println("\nTesting GGUF module...")
            include("unit/test_gguf.jl")
            println("\nTesting Inferno utilities...")
            include("unit/test_inferno_utils.jl")
        '
        ;;
    
    model)
        print_header "Running Model-Dependent Tests"
        print_status "These tests require a valid GGUF model file"
        
        if [ -z "$INFERNO_MODEL" ]; then
            print_error "INFERNO_MODEL environment variable not set"
            echo "Please set INFERNO_MODEL to your model path:"
            echo "  export INFERNO_MODEL=/path/to/model.gguf"
            exit 1
        fi
        
        if [ ! -f "$INFERNO_MODEL" ]; then
            print_error "Model file not found: $INFERNO_MODEL"
            exit 1
        fi
        
        julia --project=. runtests.jl
        ;;
    
    integration)
        print_header "Running Integration Tests"
        print_status "These tests test the full HTTP server stack"
        
        julia --project=. -e '
            using Test
            println("Testing Server auth and error handling...")
            include("unit/test_server_auth.jl")
        '
        ;;
    
    coverage)
        print_header "Running Tests with Coverage"
        print_status "Coverage data will be saved to .cov files"
        
        julia --project=. --code-coverage=user -e '
            using Test
            println("Running all unit tests with coverage...")
            include("unit/test_engine.jl")
            include("unit/test_tokenizer.jl")
            include("unit/test_gguf.jl")
            include("unit/test_inferno_utils.jl")
            include("unit/test_server_auth.jl")
        '
        ;;
    
    quick)
        print_header "Running Quick Unit Tests"
        print_status "Fast subset of unit tests for rapid feedback"
        
        julia --project=. -e '
            using Test
            @testset "Quick Unit Tests" begin
                # Only the fastest, most critical tests
                include("unit/test_engine.jl")
            end
        '
        ;;
    
    all|*)
        print_header "Running All Tests"
        print_status "Complete test suite"
        
        julia --project=. runtests.jl
        ;;
esac

# Check exit status
if [ $? -eq 0 ]; then
    print_success "All tests passed!"
else
    print_error "Some tests failed!"
    exit 1
fi