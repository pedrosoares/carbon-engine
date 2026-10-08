# Copyright 2026 Carbonis Fundatus
# SPDX-License-Identifier: Apache-2.0

"""Compile every carbon CUDA kernel with NVRTC exactly as the engine does at start-up (no GPU needed).

    python3 scripts/nvrtc_check.py /usr/local/cuda/lib64/libnvrtc.so src/kernels
"""
import ctypes, sys, pathlib
lib = ctypes.CDLL(sys.argv[1])
k = pathlib.Path(sys.argv[2])
units = {
    "quant": (["quant.cu"], 70),
    "attention": (["attention.cu"], 70),
    "attention_tc": (["attention.cu", "attention_tc.cu"], 80),
    "moe": (["quant.cu", "moe.cu"], 70),
    "gdn": (["quant.cu", "gdn.cu"], 70),
    "sampling": (["sampling.cu"], 70),
    "qsa": (["quant.cu", "qsa.cu"], 70),
}
ARCHS = [70, 75, 80, 86, 87, 89, 90, 100, 110, 120]


def compiles(src, arch):
    prog = ctypes.c_void_p()
    assert lib.nvrtcCreateProgram(ctypes.byref(prog), src, b"probe", 0, None, None) == 0
    opts = (ctypes.c_char_p * 1)(f"--gpu-architecture=compute_{arch}".encode())
    return lib.nvrtcCompileProgram(prog, 1, opts) == 0


def oldest_arch(minimum):
    """The engine's choice: the oldest virtual arch >= minimum this NVRTC accepts."""
    for a in ARCHS:
        if a >= minimum and compiles(b'extern "C" __global__ void carbon_probe() {}', a):
            return f"compute_{a}"
    return f"compute_{minimum}"


ok = True
for name, (files, minimum) in units.items():
    arch = oldest_arch(minimum)
    src = "\n".join((k / f).read_text() for f in files).encode()
    prog = ctypes.c_void_p()
    assert lib.nvrtcCreateProgram(ctypes.byref(prog), src, name.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 1)(f"--gpu-architecture={arch}".encode())
    rc = lib.nvrtcCompileProgram(prog, 1, opts)
    n = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetProgramLog(prog, log)
    entries = 0
    if rc == 0:
        lib.nvrtcGetPTXSize(prog, ctypes.byref(n))
        ptx = ctypes.create_string_buffer(n.value)
        lib.nvrtcGetPTX(prog, ptx)
        entries = ptx.value.decode().count(".entry")
    print(f"{name:13s} {arch}  {'ok' if rc == 0 else 'FAILED rc=%d' % rc}  {entries} kernels")
    if rc != 0:
        ok = False
        print(log.value.decode()[:3000])
sys.exit(0 if ok else 1)
