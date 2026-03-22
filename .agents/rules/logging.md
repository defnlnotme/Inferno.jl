---
trigger: model_decision
description: when writing log strings or print statements in the codebase
---

Don't interpolate variables in log strings. Put the variables after the string, as per julia convention.

## Use Logging Macros Instead of println

**Never use `println` for error handling or debugging output in library code.** Use Julia's logging macros:

- `@error` - For errors and exceptions
- `@warn` - For warnings and non-fatal issues  
- `@info` - For informational messages (e.g., loading progress)
- `@debug` - For debug-level output

### Error Handling with Stacktraces

When catching exceptions, use `@error` with the `exception` parameter to automatically include the stacktrace:

```julia
catch e
    @error "Operation failed" exception=(e, catch_backtrace())
    # cleanup code...
end
```

**Don't manually print stacktraces** with `println` loops. The `exception=(e, catch_backtrace())` pattern handles this properly.

### Examples

```julia
# ❌ Bad: Using println for errors
catch e
    @error "forward!: " e
    st = stacktrace(catch_backtrace())
    for line in st
        println("  ", line)
    end
end

# ✅ Good: Using logging macro with exception parameter
catch e
    @error "forward! failed" exception=(e, catch_backtrace())
end

# ❌ Bad: Using println for info messages
println("Loading model: $path")

# ✅ Good: Using @info for informational messages
@info "Loading model" path
```
