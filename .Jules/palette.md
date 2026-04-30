## 2025-05-15 - [Graceful CLI Interrupts & Feedback]
**Learning:** For interactive Julia CLI tools, specifically LLM engines where operations like model loading or prompt input can be interrupted, wrapping `readline()` and `load_model()` in `try-catch` for `EOFError` and `InterruptException` is essential for a polished UX that avoids raw stack traces. Providing immediate visual feedback (e.g., dots) during long-running weight loading makes the app feel responsive even before inference starts.
**Action:** Use standard `try-catch` blocks for `EOFError` and `InterruptException` in all CLI entry points and add incremental progress printing in `Loader.jl`.

## 2025-05-16 - [Robust TTY Feedback in Custom Line Editors]
**Learning:** When implementing custom line editors or streaming feedback in Julia, `Base.TTY` might be wrapped by `TTYTerminal`. Robustly detecting a TTY requires checking both `isa(io, Base.TTY)` and `hasfield(typeof(io), :io) && isa(io.io, Base.TTY)`. Additionally, using `\e[2K` for line clearing instead of space-padding prevents visual artifacts during rapid terminal refreshes.
**Action:** Use the `hasfield` check for TTY-specific feedback (like thinking indicators) and prefer ANSI `\e[2K` for clearing lines in `refresh_line` implementations.
