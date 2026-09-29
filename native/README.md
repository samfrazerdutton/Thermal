# native/

C++ and CUDA code: kernel lab, native runtime pieces, and micro-benchmarks.

| Directory | Purpose |
|---|---|
| `cuda/` | CUDA kernel lab: vector_add (naive + grid-stride), reduction (naive atomic + shared-memory), matmul (naive + tiled), plus `bench_main.cu`, a timing harness that verifies correctness against a CPU reference before reporting any number |
| `cpp/` | Host-side C++ runtime (not yet populated — NVML wrappers, process harness land alongside Phase 12) |
| `benchmarks/` | Reserved for aggregated benchmark output; per-run JSON currently comes straight from `bench_main.cu`'s stdout |

## Building

```powershell
powershell -File scripts/build-native.ps1
```

Then either run the binary directly or through the CLI:

```bash
thermal kernel list
thermal kernel run matmul_tiled --n 1024
```

## Windows build note

`cl.exe` is only on `PATH` inside a Visual Studio Developer Command Prompt (VS 2019
BuildTools, `VC.Tools.x86.x64` component — confirmed via `vswhere`). More importantly,
**the CUDA toolkit on the reference machine has no Visual Studio integration
installed**, so CMake's default `Visual Studio 16 2019` generator cannot compile `.cu`
files at all (`check_language(CUDA)` returns `NOTFOUND` under that generator — verified,
see `docs/environment-report.md`). The real fix, also verified: configure with the
Ninja generator after loading the MSVC environment (`vcvarsall.bat x64`) — Ninja invokes
`nvcc`/`cl` directly and doesn't need the missing VS integration files.
`scripts/build-native.ps1` automates exactly this.
