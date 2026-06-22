! brainstem_cli.f90 — Fortran Brainstem CLI (Daemon Mode)
! Compile: gfortran -O3 -march=native -flto -funroll-loops -ffast-math -fopenmp
!          brainstem.f90 brainstem_cli.f90 -o brainstem_cli.exe
!
! Modes:
!   echo "2.0 0.0 ... 0.0" | brainstem_cli.exe           → one-shot
!   echo "2.0 0.0 ... 0.0" | brainstem_cli.exe --daemon  → persistent stdin loop
!   brainstem_cli.exe --daemon < features.txt             → batch mode
!
! Daemon mode: INIT ONCE, classify many times. Eliminates the ~5ms
! initialization overhead per call for standby monitoring.

program brainstem_cli
  use brainstem_mod, only: do_full_init, do_classify
  implicit none
  integer, parameter :: N_DIMS = 22
  real(kind=8) :: features(N_DIMS)
  integer :: region_id, seed, io_status, count
  real(kind=8) :: confidence, difficulty
  character(len=32) :: arg
  logical :: daemon_mode

  daemon_mode = .false.

  ! Check for --daemon flag
  if (command_argument_count() >= 1) then
    call get_command_argument(1, arg)
    if (trim(arg) == '--daemon') daemon_mode = .true.
  end if

  seed = 42
  call do_full_init(seed)

  if (daemon_mode) then
    ! ── Daemon Mode: init once, loop forever reading stdin ──
    count = 0
    do
      read(*, *, iostat=io_status) features(1:N_DIMS)
      if (io_status /= 0) exit  ! EOF or error → exit

      call do_classify(features, region_id, confidence, difficulty)
      write(*, '(I0,1X,F9.6,1X,F9.6)') region_id, confidence, difficulty
      count = count + 1
    end do
  else
    ! ── One-shot Mode (backward compatible) ──
    read(*, *) features(1:N_DIMS)
    call do_classify(features, region_id, confidence, difficulty)
    write(*, '(I0,1X,F9.6,1X,F9.6)') region_id, confidence, difficulty
  end if
end program brainstem_cli
