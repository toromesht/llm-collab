! brainstem_cli.f90 — Fortran Brainstem CLI
! Compile: gfortran -O3 -march=native -flto -funroll-loops -ffast-math -fopenmp -m64
!          brainstem.f90 brainstem_cli.f90 -o brainstem_cli.exe
! Usage: echo "2.0 0.0 0.0 ... 0.0" | brainstem_cli.exe [seed]

program brainstem_cli
  use brainstem_mod, only: do_full_init, do_classify
  implicit none
  integer, parameter :: N_DIMS = 22
  real(kind=8) :: features(N_DIMS)
  integer :: region_id, seed
  real(kind=8) :: confidence, difficulty

  seed = 42
  call do_full_init(seed)

  ! Read all 22 floats from one line
  read(*, *) features(1:N_DIMS)

  call do_classify(features, region_id, confidence, difficulty)
  write(*, '(I0,1X,F9.6,1X,F9.6)') region_id, confidence, difficulty
end program brainstem_cli
