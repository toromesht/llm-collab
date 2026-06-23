# ═══════════════════════════════════════════════════════════════
# SynapseFlow Fortran-MPI Build
# ═══════════════════════════════════════════════════════════════
#
# Requirements: OpenMPI or MPICH, gfortran >= 10
#
# Usage:
#   make                    # Build MPI brainstem executable
#   make lib                # Build shared library for Python ctypes
#   make clean              # Remove build artifacts
#   mpirun -np 4 ./brainstem_mpi.exe   # Run with 4 MPI processes
#
# Architecture:
#   mpi_utils.f90       — MPI communication layer (scatter/gather/reduce)
#   hd_encode.f90       — HD encoding (already exists, used as-is)
#   parallel_ops.f90    — OpenMP parallel ops (already exists, used as-is)
#   brainstem_mpi.f90   — Main MPI orchestrator
# ═══════════════════════════════════════════════════════════════

FC      = mpif90
FFLAGS  = -O3 -fopenmp -cpp -ffree-line-length-none -std=f2018
LDFLAGS = -fopenmp

# Source files (hd_encode.f90 and parallel_ops.f90 are in parent dir)
SRC = ../hd_encode.f90 ../parallel_ops.f90 mpi_utils.f90 brainstem_mpi.f90
OBJ = $(SRC:.f90=.o)

TARGET  = brainstem_mpi.exe
LIB     = libbrainstem_mpi.so

.PHONY: all lib clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(FC) $(FFLAGS) -o $@ $(SRC) $(LDFLAGS)
	@echo "[OK] Built $(TARGET)"

lib: $(SRC)
	$(FC) $(FFLAGS) -shared -fPIC -o $(LIB) $(SRC) $(LDFLAGS)
	@echo "[OK] Built $(LIB)"

clean:
	rm -f *.o *.mod *.so *.exe

# Dependencies
mpi_utils.o: mpi_utils.f90
brainstem_mpi.o: brainstem_mpi.f90 mpi_utils.o
