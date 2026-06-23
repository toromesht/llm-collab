# SynapseFlow Build Guide

## Prerequisites

All platforms need:
- **Python 3.11+** (64-bit) with `pybind11` installed
- **CMake 3.20+**

### Windows (MinGW)

```bash
# 1. Install MSYS2 from https://www.msys2.org/
# 2. Open MSYS2 UCRT64 terminal and install tools:
pacman -S mingw-w64-ucrt-x86_64-gcc \
          mingw-w64-ucrt-x86_64-gcc-fortran \
          mingw-w64-ucrt-x86_64-cmake \
          mingw-w64-ucrt-x86_64-ninja \
          mingw-w64-ucrt-x86_64-openmp

# 3. Install Python dependencies
pip install pybind11 numpy

# 4. Add UCRT64 bin to PATH (e.g., C:\msys64\ucrt64\bin)
```

### Linux (GCC)

```bash
# Ubuntu/Debian
sudo apt install build-essential gfortran cmake libopenmpi-dev python3-dev

# Fedora
sudo dnf install gcc-gfortran cmake openmpi-devel python3-devel

# Then:
pip install pybind11 numpy
```

### macOS (Homebrew)

```bash
brew install gcc cmake libomp python@3.11
pip install pybind11 numpy
```

## Build

```bash
cd collab-cloud
rm -rf build && mkdir build

# Configure (auto-detects compilers from PATH)
cmake -B build -G "MinGW Makefiles" \
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release -j$(nproc)
```

### Specifying compilers manually

```bash
cmake -B build \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_Fortran_COMPILER=gfortran \
  -DCMAKE_BUILD_TYPE=Release
```

## Output

```
build/lib/
├── synapse_router.cp311-win_amd64.pyd   # C++ routing engine (Python module)
└── libsynapse_encode.a                  # Fortran HD/SDM encoder (static lib)
```

## Verify

```bash
python -c "
import sys; sys.path.insert(0, 'build/lib')
import synapse_router
engine = synapse_router.RouterEngine()
print('C++ Router:', engine.stats())
"
```

## Troubleshooting

### "DLL load failed" on Windows
The C++ module needs MinGW runtime DLLs. They are auto-discovered from g++ in PATH.
Make sure the MinGW `bin/` directory is in your system PATH.

### "c_f_pointer_ undefined reference" (MinGW)
This is a known MinGW gfortran issue with `iso_c_binding` intrinsics.
The Fortran library is built as **static** (.a) to work around this.
For Python access, the static lib needs a C wrapper DLL (not yet implemented).

### Compiler not found
CMake searches PATH for `gcc`, `g++`, `gfortran`.
On Windows with MSYS2, use the UCRT64 or MINGW64 terminal.

### Python version mismatch
The build targets Python 3.11 by default. Edit `CMakeLists.txt` to change:
```cmake
find_package(Python 3.12 ...)  # or your version
```
