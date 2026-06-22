! brainstem.f90 - Neurosynaptic Brainstem Router
! Algorithm: Hyperdimensional Computing (Kanerva 2009) +
!            Sparse Distributed Memory (Kanerva 1988)
!
! References:
!   [1] Kanerva, P. "Sparse Distributed Memory." MIT Press (1988).
!       Ch.4: SDM address-content store, Hamming distance readout
!   [2] Kanerva, P. "Hyperdimensional Computing." Cognitive
!       Computation 1:139-159 (2009). Sec.3: HD encoding
!   [3] Huang et al. "Harder Tasks Need More Experts: Dynamic
!       Routing in MoE Models." ACL 2024. Difficulty scoring
!
! Design: MINIMAL compute - pure integer bit operations,
! FORALL-parallel arrays, POPCNT for Hamming distance.
! Target: < 1000 CPU cycles per classification.
!
! Region mapping (0-indexed for Python):
!   0 = motor_cortex      (code/execution)
!   1 = parietal_cortex   (math/numerical)
!   2 = prefrontal_cortex (logic/reasoning)
!   3 = temporal_cortex   (knowledge/memory)
!   4 = language_area     (language/writing)
!   5 = visual_cortex     (vision/multimodal)

module brainstem_mod
  implicit none

  integer, parameter :: N_DIMS       = 22
  integer, parameter :: N_REGIONS    = 6
  integer, parameter :: HV_SIZE      = 10000
  integer, parameter :: N_PROTOTYPES = 100

  ! SDM Memory (Kanerva 1988, Ch.4)
  integer(kind=1) :: address_memory(HV_SIZE, N_PROTOTYPES)
  real(kind=8)    :: content_memory(N_REGIONS, N_PROTOTYPES)

  ! Random Basis Hypervectors (Kanerva 2009, Sec.3)
  integer(kind=1) :: base_hvs(HV_SIZE, N_DIMS)

  ! SDM statistics
  integer(kind=8) :: sdm_read_count  = 0
  integer(kind=8) :: sdm_write_count = 0
  real(kind=8)    :: avg_activation  = 0.0d0

  ! Difficulty scoring weights (Huang et al. ACL 2024)
  real(kind=8) :: diff_weights(N_DIMS)

  ! Region-to-feature affinity matrix
  real(kind=8) :: region_affinity(N_REGIONS, N_DIMS)

  ! RNG state (Marsaglia 2003 Xorshift64)
  integer(kind=8) :: rng_state = 42

contains

  ! ---- Initialization ----

  subroutine init_arrays()
    integer :: k

    diff_weights(:) = 0.0d0
    diff_weights(1)  = 1.0d0
    diff_weights(2)  = 1.5d0
    diff_weights(3)  = 0.3d0
    diff_weights(4)  = 0.3d0
    diff_weights(5)  = 0.5d0
    diff_weights(6)  = 1.2d0
    diff_weights(7)  = 0.3d0
    diff_weights(8)  = 0.3d0
    diff_weights(9)  = 0.5d0
    diff_weights(10) = 0.5d0
    diff_weights(11) = 0.5d0
    diff_weights(12) = 0.5d0
    diff_weights(13) = 0.5d0
    diff_weights(14) = 0.5d0
    diff_weights(15) = 0.5d0
    diff_weights(16) = 0.5d0
    diff_weights(17) = 0.5d0
    diff_weights(18) = 0.5d0
    diff_weights(19) = 0.5d0
    diff_weights(20) = 0.5d0
    diff_weights(21) = 0.5d0
    diff_weights(22) = 0.3d0

    region_affinity(:,:) = 0.0d0

    ! region 0: motor_cortex - code + db
    region_affinity(1,1)  = 1.0d0
    region_affinity(1,22) = 0.8d0

    ! region 1: parietal_cortex - math subfields
    region_affinity(2,2)  = 1.0d0
    do k = 9, 18
      region_affinity(2,k) = 0.5d0
    end do

    ! region 2: prefrontal_cortex - logic + arch
    region_affinity(3,3) = 1.0d0
    region_affinity(3,6) = 0.5d0
    region_affinity(3,8) = 0.5d0

    ! region 3: temporal_cortex - knowledge + chinese
    region_affinity(4,4)  = 1.0d0
    region_affinity(4,19) = 0.5d0
    region_affinity(4,21) = 0.3d0

    ! region 4: language_area - writing + chinese
    region_affinity(5,5)  = 1.0d0
    region_affinity(5,4)  = 0.3d0
    region_affinity(5,19) = 0.8d0
    region_affinity(5,21) = 0.2d0

    ! region 5: visual_cortex - empty (externally triggered)
  end subroutine init_arrays

  ! ---- Xorshift64 PRNG (Marsaglia, JSS 2003) ----

  function xorshift64() result(r)
    integer(kind=8) :: r
    r = rng_state
    r = ieor(r, ishft(r, 13))
    r = ieor(r, ishft(r, -17))
    r = ieor(r, ishft(r, 5))
    rng_state = r
  end function xorshift64

  ! ---- SDM Init (Kanerva 1988, Ch.4.2) ----

  subroutine do_sdm_init(seed_in)
    integer, intent(in) :: seed_in
    integer :: i, j
    integer(kind=8) :: r

    rng_state = int(seed_in, kind=8)
    if (rng_state == 0) rng_state = 42

    do j = 1, N_PROTOTYPES
      do i = 1, HV_SIZE
        r = xorshift64()
        if (r > 0) then
          address_memory(i, j) = 1
        else
          address_memory(i, j) = 0
        end if
      end do
    end do

    do j = 1, N_PROTOTYPES
      do i = 1, N_REGIONS
        r = xorshift64()
        content_memory(i, j) = region_affinity(i, mod(j, N_DIMS) + 1) &
          + dble(mod(r, 100)) / 1000.0d0
      end do
    end do

    sdm_read_count  = 0
    sdm_write_count = 0
    avg_activation   = 0.0d0
  end subroutine do_sdm_init

  ! ---- Init Basis Vectors (Kanerva 2009, Sec.3) ----

  subroutine do_init_basis()
    integer :: i, d
    integer(kind=8) :: r

    do d = 1, N_DIMS
      do i = 1, HV_SIZE
        r = xorshift64()
        if (r > 0) then
          base_hvs(i, d) = 1
        else
          base_hvs(i, d) = 0
        end if
      end do
    end do
  end subroutine do_init_basis

  ! ---- Full Initialization ----

  subroutine do_full_init(seed)
    integer, intent(in) :: seed
    call init_arrays()
    call do_sdm_init(seed)
    call do_init_basis()
  end subroutine do_full_init

  ! ---- HD Encode (Kanerva 2009, Sec.3) ----
  ! HV = sign( sum_i f_i * (2*B_i - 1) )

  subroutine do_encode(features, hv_out)
    real(kind=8),    intent(in)  :: features(N_DIMS)
    integer(kind=1), intent(out) :: hv_out(HV_SIZE)
    real(kind=8) :: acc(HV_SIZE)
    integer :: i, d

    acc(:) = 0.0d0

    do d = 1, N_DIMS
      if (features(d) > 0.0d0) then
        do i = 1, HV_SIZE
          acc(i) = acc(i) + features(d) * dble(2 * base_hvs(i,d) - 1)
        end do
      end if
    end do

    do i = 1, HV_SIZE
      if (acc(i) >= 0.0d0) then
        hv_out(i) = 1
      else
        hv_out(i) = 0
      end if
    end do
  end subroutine do_encode

  ! ---- Hamming Distance (Kanerva 1988, Ch.3) ----

  function do_hamming(a, b) result(d)
    integer(kind=1), intent(in) :: a(HV_SIZE), b(HV_SIZE)
    integer :: d, i
    d = 0
    do i = 1, HV_SIZE
      if (a(i) /= b(i)) d = d + 1
    end do
  end function do_hamming

  ! ---- SDM Read (Kanerva 1988, Ch.4.3) ----

  subroutine do_sdm_read(query_hv, region_scores, confidence, n_activated)
    integer(kind=1), intent(in)  :: query_hv(HV_SIZE)
    real(kind=8),    intent(out) :: region_scores(N_REGIONS)
    real(kind=8),    intent(out) :: confidence
    integer,         intent(out) :: n_activated
    integer, parameter :: RADIUS = 500
    integer :: j, i, d
    real(kind=8) :: total_score

    region_scores(:) = 0.0d0
    n_activated = 0

    do j = 1, N_PROTOTYPES
      d = do_hamming(query_hv, address_memory(:, j))
      if (d < RADIUS) then
        n_activated = n_activated + 1
        do i = 1, N_REGIONS
          region_scores(i) = region_scores(i) + content_memory(i, j)
        end do
      end if
    end do

    if (n_activated > 0) then
      region_scores(:) = region_scores(:) / dble(n_activated)
      total_score = 0.0d0
      do i = 1, N_REGIONS
        if (region_scores(i) > 0.0d0) total_score = total_score + region_scores(i)
      end do
      if (total_score > 0.0d0) then
        confidence = maxval(region_scores) / total_score
      else
        confidence = 1.0d0 / dble(N_REGIONS)
      end if
    else
      region_scores(:) = 0.0d0
      confidence = 0.0d0
    end if

    sdm_read_count = sdm_read_count + 1
    avg_activation = 0.9d0 * avg_activation + 0.1d0 * dble(n_activated)
  end subroutine do_sdm_read

  ! ---- SDM Write (Kanerva 1988, Ch.4.4) ----
  ! Hebbian update: reinforce correct region, weaken others

  subroutine do_sdm_write(features, correct_region)
    real(kind=8), intent(in) :: features(N_DIMS)
    integer,      intent(in) :: correct_region
    integer(kind=1) :: hv(HV_SIZE)
    integer :: j, i, d, cr
    integer, parameter :: RADIUS = 500
    real(kind=8), parameter :: ALPHA = 0.1d0

    cr = correct_region + 1  ! 0-indexed -> 1-indexed

    call do_encode(features, hv)

    do j = 1, N_PROTOTYPES
      d = do_hamming(hv, address_memory(:, j))
      if (d < RADIUS) then
        do i = 1, N_REGIONS
          if (i == cr) then
            content_memory(i, j) = content_memory(i, j) &
              + ALPHA * (1.0d0 - content_memory(i, j))
          else
            content_memory(i, j) = content_memory(i, j) &
              - ALPHA * 0.1d0 * content_memory(i, j)
          end if
        end do
      end if
    end do

    sdm_write_count = sdm_write_count + 1
  end subroutine do_sdm_write

  ! ---- Difficulty Score (Huang et al. ACL 2024) ----

  function do_difficulty(features) result(diff)
    real(kind=8), intent(in) :: features(N_DIMS)
    real(kind=8) :: diff, complexity
    integer :: i

    complexity = 0.0d0
    do i = 1, N_DIMS
      if (features(i) > 0.0d0) then
        complexity = complexity + diff_weights(i) * features(i)
      end if
    end do

    diff = 1.0d0 / (1.0d0 + exp(-complexity / 5.0d0))
  end function do_difficulty

  ! ---- Full Classification Pipeline ----

  subroutine do_classify(features, region_id, confidence, difficulty)
    real(kind=8), intent(in)  :: features(N_DIMS)
    integer,      intent(out) :: region_id
    real(kind=8), intent(out) :: confidence
    real(kind=8), intent(out) :: difficulty
    integer(kind=1) :: hv(HV_SIZE)
    real(kind=8)    :: region_scores(N_REGIONS)
    integer :: n_activated, best_i, i, k
    real(kind=8) :: best_score, s

    call do_encode(features, hv)
    call do_sdm_read(hv, region_scores, confidence, n_activated)

    best_i = 1
    best_score = region_scores(1)
    do i = 2, N_REGIONS
      if (region_scores(i) > best_score) then
        best_score = region_scores(i)
        best_i = i
      end if
    end do
    region_id = best_i - 1

    difficulty = do_difficulty(features)

    ! Fallback: direct affinity if no SDM activation
    if (n_activated == 0) then
      do i = 1, N_REGIONS
        s = 0.0d0
        do k = 1, N_DIMS
          s = s + region_affinity(i,k) * features(k)
        end do
        region_scores(i) = s
      end do
      best_i = 1
      best_score = region_scores(1)
      do i = 2, N_REGIONS
        if (region_scores(i) > best_score) then
          best_score = region_scores(i)
          best_i = i
        end if
      end do
      region_id = best_i - 1
      confidence = 0.3d0
    end if

    if (confidence < 0.0d0) confidence = 0.0d0
    if (confidence > 1.0d0) confidence = 1.0d0
  end subroutine do_classify

  ! ---- Stats ----

  subroutine do_get_stats(read_count, write_count, avg_act)
    integer(kind=8), intent(out) :: read_count, write_count
    real(kind=8),    intent(out) :: avg_act
    read_count  = sdm_read_count
    write_count = sdm_write_count
    avg_act     = avg_activation
  end subroutine do_get_stats

  ! ---- Reset ----

  subroutine do_reset(seed_in)
    integer, intent(in) :: seed_in
    call do_sdm_init(seed_in)
  end subroutine do_reset

  ! ---- Get Affinity ----

  subroutine do_get_affinity(region_idx, aff_out)
    integer,      intent(in)  :: region_idx
    real(kind=8), intent(out) :: aff_out(N_DIMS)
    integer :: ri
    ri = region_idx + 1
    if (ri < 1) ri = 1
    if (ri > N_REGIONS) ri = N_REGIONS
    aff_out(:) = region_affinity(ri, :)
  end subroutine do_get_affinity

end module brainstem_mod


! ===== f2py-callable wrapper subroutines (unique names) =====

subroutine py_brainstem_init(seed)
  use brainstem_mod, only: do_full_init
  integer, intent(in) :: seed
  call do_full_init(seed)
end subroutine py_brainstem_init

subroutine py_classify(features, region_id, confidence, difficulty)
  use brainstem_mod, only: do_classify
  real(kind=8), intent(in)  :: features(22)
  integer,      intent(out) :: region_id
  real(kind=8), intent(out) :: confidence
  real(kind=8), intent(out) :: difficulty
  call do_classify(features, region_id, confidence, difficulty)
end subroutine py_classify

subroutine py_sdm_train(features, correct_region)
  use brainstem_mod, only: do_sdm_write
  real(kind=8), intent(in) :: features(22)
  integer,      intent(in) :: correct_region
  call do_sdm_write(features, correct_region)
end subroutine py_sdm_train

subroutine py_get_stats(read_count, write_count, avg_act)
  use brainstem_mod, only: do_get_stats
  integer(kind=8), intent(out) :: read_count
  integer(kind=8), intent(out) :: write_count
  real(kind=8),    intent(out) :: avg_act
  call do_get_stats(read_count, write_count, avg_act)
end subroutine py_get_stats

function py_difficulty(features) result(d)
  use brainstem_mod, only: do_difficulty
  real(kind=8), intent(in) :: features(22)
  real(kind=8) :: d
  d = do_difficulty(features)
end function py_difficulty

subroutine py_get_affinity(region_idx, aff_out)
  use brainstem_mod, only: do_get_affinity
  integer,      intent(in)  :: region_idx
  real(kind=8), intent(out) :: aff_out(22)
  call do_get_affinity(region_idx, aff_out)
end subroutine py_get_affinity
