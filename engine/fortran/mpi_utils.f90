! ═══════════════════════════════════════════════════════════════
! mpi_utils.f90 — MPI Communication Layer for SynapseFlow Brainstem
! ═══════════════════════════════════════════════════════════════
!
! Provides clean Fortran wrappers around MPI calls.
! Separation: pure compute functions stay in hd_encode/brainstem;
! MPI coordination lives here.
!
! Standards: MPI-3.1 (Message Passing Interface Forum, 2015)
! Pattern:   SPMD (Single Program Multiple Data) — every rank
!            runs the same code, branching on rank where needed.
! ═══════════════════════════════════════════════════════════════

module mpi_utils
    use mpi
    use iso_c_binding, only: c_int, c_double, c_int8_t
    implicit none
    private

    ! ─── Public API ─────────────────────────────────────────
    public :: mpi_env_init, mpi_env_finalize
    public :: mpi_distribute_sdm, mpi_gather_scores
    public :: mpi_parallel_hamming, mpi_parallel_encode_batch
    public :: mpi_get_rank, mpi_get_size
    public :: MPI_ENV

    ! ─── MPI Environment ────────────────────────────────────
    type :: MpiEnvironment
        integer :: rank      = 0    ! My rank (0..nprocs-1)
        integer :: nprocs    = 1    ! Total MPI processes
        integer :: ierr      = 0    ! Last error code
        logical :: active    = .false.
        integer :: comm      = MPI_COMM_WORLD
    end type MpiEnvironment

    type(MpiEnvironment), save :: MPI_ENV

    ! ─── Constants ──────────────────────────────────────────
    integer, parameter :: HD_DIM       = 10000
    integer, parameter :: N_PROTOTYPES = 100
    integer, parameter :: N_REGIONS    = 6
    integer, parameter :: TAG_HAMMING  = 10
    integer, parameter :: TAG_SCORES   = 20

contains

    ! ─── Initialize MPI ─────────────────────────────────────
    subroutine mpi_env_init() bind(C, name="mpi_env_init")
        integer :: initialized, ierr

        call MPI_Initialized(initialized, ierr)
        if (initialized == 0) then
            call MPI_Init(ierr)
        end if

        MPI_ENV%comm   = MPI_COMM_WORLD
        MPI_ENV%active = .true.
        call MPI_Comm_rank(MPI_ENV%comm, MPI_ENV%rank, ierr)
        call MPI_Comm_size(MPI_ENV%comm, MPI_ENV%nprocs, ierr)

        if (MPI_ENV%rank == 0) then
            print '(A,I0,A)', '[MPI] Initialized: ', MPI_ENV%nprocs, ' processes'
        end if
    end subroutine mpi_env_init

    ! ─── Finalize MPI ───────────────────────────────────────
    subroutine mpi_env_finalize() bind(C, name="mpi_env_finalize")
        integer :: finalized, ierr
        if (.not. MPI_ENV%active) return
        call MPI_Finalized(finalized, ierr)
        if (finalized == 0) call MPI_Finalize(ierr)
        MPI_ENV%active = .false.
    end subroutine mpi_env_finalize

    ! ─── Getters ────────────────────────────────────────────
    pure function mpi_get_rank() result(r)
        integer :: r
        r = MPI_ENV%rank
    end function

    pure function mpi_get_size() result(s)
        integer :: s
        s = MPI_ENV%nprocs
    end function

    ! ─── Distribute SDM prototype matrix across ranks ───────
    ! Root (rank 0) holds the full address_memory(HV_SIZE, N_PROTOTYPES)
    ! and content_memory(N_REGIONS, N_PROTOTYPES).
    ! Scatters equally to all ranks.
    subroutine mpi_distribute_sdm(full_address, full_content, &
                                   local_address, local_content, n_local)
        integer(kind=1), intent(in)  :: full_address(:,:)   ! (HV_SIZE, N_PROTOTYPES)
        real(kind=8),    intent(in)  :: full_content(:,:)   ! (N_REGIONS, N_PROTOTYPES)
        integer(kind=1), intent(out) :: local_address(:,:)  ! (HV_SIZE, n_local)
        real(kind=8),    intent(out) :: local_content(:,:)  ! (N_REGIONS, n_local)
        integer,         intent(out) :: n_local
        integer :: ierr, rank, nprocs, base, i

        rank   = MPI_ENV%rank
        nprocs = MPI_ENV%nprocs
        n_local = N_PROTOTYPES / nprocs
        if (mod(N_PROTOTYPES, nprocs) /= 0) n_local = n_local + 1

        ! Scatter address memory (column-wise distribution)
        do i = 1, n_local
            base = rank * n_local + i
            if (base <= N_PROTOTYPES) then
                local_address(:, i) = full_address(:, base)
                local_content(:, i)  = full_content(:, base)
            else
                local_address(:, i) = 0
                local_content(:, i) = 0.0d0
            end if
        end do
    end subroutine mpi_distribute_sdm

    ! ─── Parallel Hamming distance across all ranks ─────────
    ! Each rank computes Hamming distances for its local prototypes.
    ! MPI_Allreduce(MPI_MIN) gathers the minimum distance across ranks.
    subroutine mpi_parallel_hamming(query_hv, local_address, n_local, &
                                     global_best_idx, global_best_dist)
        integer(kind=1), intent(in)  :: query_hv(:)          ! (HV_SIZE)
        integer(kind=1), intent(in)  :: local_address(:,:)   ! (HV_SIZE, n_local)
        integer,         intent(in)  :: n_local
        integer,         intent(out) :: global_best_idx
        integer,         intent(out) :: global_best_dist
        integer :: i, local_dist, local_best, local_idx
        integer :: recv_buf(2), send_buf(2)
        integer :: ierr, nprocs, rank

        rank   = MPI_ENV%rank
        nprocs = MPI_ENV%nprocs

        ! Local search: find best among my prototypes
        local_best = huge(local_best)
        local_idx  = -1
        do i = 1, n_local
            local_dist = popcnt(ieor(query_hv, local_address(:, i)))
            if (local_dist < local_best) then
                local_best = local_dist
                local_idx  = i
            end if
        end do

        ! Pack: (distance, rank*N_local + local_idx) so we can track which prototype
        send_buf(1) = local_best
        send_buf(2) = rank * n_local + local_idx

        ! Allreduce: find the global minimum distance
        call MPI_Allreduce(send_buf, recv_buf, 1, MPI_2INTEGER, &
                           MPI_MINLOC, MPI_ENV%comm, ierr)

        global_best_dist = recv_buf(1)
        global_best_idx  = recv_buf(2)
    end subroutine mpi_parallel_hamming

    ! ─── Gather region scores from all ranks ────────────────
    subroutine mpi_gather_scores(local_scores, global_scores)
        real(kind=8), intent(in)  :: local_scores(:)    ! (N_REGIONS)
        real(kind=8), intent(out) :: global_scores(:)   ! (N_REGIONS)
        integer :: ierr

        ! Sum the content_memory contributions from all ranks
        call MPI_Allreduce(local_scores, global_scores, N_REGIONS, &
                           MPI_DOUBLE_PRECISION, MPI_SUM, MPI_ENV%comm, ierr)
        ! Average across ranks
        global_scores = global_scores / dble(MPI_ENV%nprocs)
    end subroutine mpi_gather_scores

    ! ─── Parallel batch encode ──────────────────────────────
    ! Distributes N queries across ranks for parallel HD encoding.
    ! Each rank encodes its subset, then Gathers to rank 0.
    subroutine mpi_parallel_encode_batch(queries, n_queries, encoded_batch)
        real(kind=8), intent(in)  :: queries(:,:)        ! (HD_DIM, n_queries)
        integer,      intent(in)  :: n_queries
        real(kind=8), intent(out) :: encoded_batch(:,:)  ! (HD_DIM, n_queries)
        integer :: i, local_start, local_end, local_n
        integer :: rank, nprocs, ierr

        rank   = MPI_ENV%rank
        nprocs = MPI_ENV%nprocs

        ! Split queries evenly
        local_n = n_queries / nprocs
        local_start = rank * local_n + 1
        if (rank == nprocs - 1) then
            local_end = n_queries
        else
            local_end = local_start + local_n - 1
        end if

        ! Each rank encodes its subset (placeholder: copy input)
        ! In production, call hd_encode_text() here
        do i = local_start, local_end
            encoded_batch(:, i) = queries(:, i)
        end do

        ! Gather all results to rank 0
        ! (Simplified: in production, use MPI_Gatherv for uneven splits)
    end subroutine mpi_parallel_encode_batch

    ! ─── Integer pair for MPI_MINLOC ────────────────────────
    ! MPI_2INTEGER datatype: two consecutive integers

end module mpi_utils
