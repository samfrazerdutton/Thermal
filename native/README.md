# native/

C++ and CUDA code: kernel lab, native runtime pieces, and micro-benchmarks. Lands in
Phase 7.

| Directory | Purpose |
|---|---|
| `cuda/` | CUDA kernels (vector add, reduction, matmul, memory-bandwidth probes) |
| `cpp/` | Host-side C++ runtime (CUDA event timing, NVML wrappers, process harness) |
| `benchmarks/` | Machine-readable benchmark output consumed by `core/experiments` |

## Windows build note

On the reference development machine, `cl.exe` is only on `PATH` inside a Visual
Studio Developer Command Prompt (VS 2019 BuildTools, `VC.Tools.x86.x64` component is
installed — confirmed via `vswhere`, see `docs/environment-report.md`). CMake presets
added in Phase 7 will either require that shell or drive the MSVC toolchain directly
via the `-G "Visual Studio 16 2019"` generator, rather than assuming `cl`/`g++` are on
a plain `PATH`.
