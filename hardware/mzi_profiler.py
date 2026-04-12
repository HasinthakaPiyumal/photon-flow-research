"""
hardware/mzi_profiler.py

MZI hardware simulation: SVD decomposition, phase quantization,
optical loss, detector noise, thermal crosstalk.

Part 1 -- SVD to MZI phases + quantization:
    W = U Sigma V^T  (Shen 2017).  Each unitary (U, V) is decomposed
    into N(N-1)/2 Givens rotations (Reck 1994), one per physical MZI.
    Phases are quantized to 4-6 bit precision (Ning 2024).

Part 2 -- Hardware imperfection simulation:
    After setting the MZI phases on chip, four noise sources degrade
    the effective weight matrix:

    (a) Phase encoding noise (Shen 2017, sigma_phi ~ 5e-3 rad):
        Each phase shifter has a small random offset from its target
        value.  Modelled as independent additive Gaussian per phase.

    (b) Thermal crosstalk (Shen 2017, Ning 2024, sigma_t = 0.01):
        Heating one phase shifter warms its neighbours, shifting their
        phases.  Modelled as correlated additive Gaussian -- each phase
        receives a fraction of its neighbours' noise.

    (c) Optical loss (Ning 2024, 0.1 dB per MZI stage):
        Light power decreases as it passes through beamsplitters.
        Cumulative: n stages x 0.1 dB.  Applied as uniform attenuation
        of the singular values (signal amplitude scales with
        10^(-total_dB / 20)).

    (d) Detector noise (Shen 2017, sigma_s = 0.02):
        Photodetectors add shot noise to the output signal.
        Modelled as additive Gaussian on the output vector.

    The full simulate() pipeline chains all five steps:
        decompose -> quantize -> phase noise -> thermal crosstalk
        -> reconstruct -> optical loss -> (detector noise at inference)

Simulation framework (Ning 2024, Section "Simulation framework"):
    1. Decompose weight matrix into MZI phases (SVD + Reck).
    2. Apply phase quantization (4-6 bit DAC precision).
    3. Add optical loss (0.1 dB/stage cumulative attenuation).
    4. Add detector noise (additive Gaussian at output).
    5. Add thermal crosstalk (correlated noise on adjacent phases).

References:
    Shen et al., "Deep Learning with Coherent Nanophotonic Circuits,"
    Nature Photonics, 2017.
    (SVD: W = U Sigma V^dagger.  Noise model: sigma_phi ~ 5e-3 rad,
    sigma_D ~ 0.1%, thermal crosstalk dominant.  MZI fidelity 99.8%.)

    Reck et al., "Experimental Realization of Any Discrete Unitary
    Operator," Phys. Rev. Lett., 1994.
    (NxN unitary -> N(N-1)/2 MZI decomposition.)

    Ning et al., "Photonic-Electronic Integrated Circuits for High-
    Performance Computing and AI Accelerators," J. Lightwave Technol., 2024.
    (0.1 dB/stage, 4-6 bit precision, sigma_s = 0.01-0.03 detector
    noise, thermal crosstalk coefficients decay with distance.)

    Ning et al., "StrC-ONN: Hardware-Efficient Structured Compression
    for Optical Neural Networks," Optica, 2025.
    (Independent validation: phase quantization + optical crosstalk +
    insertion loss + thermal drift modelled during training.)
"""

import math

import numpy as np


# =====================================================================
# Part 1 -- Givens rotation decomposition (Reck-style)
# =====================================================================

def decompose_unitary(U):
    """Decompose NxN real orthogonal matrix into Givens rotations.

    Each Givens rotation = one physical MZI on chip (Reck 1994).
    Zeroes sub-diagonal elements column-by-column.

    Args:
        U: (N, N) real orthogonal matrix.

    Returns:
        thetas: list of (row_i, row_j, theta) -- one per MZI.
        diag:   (N,) diagonal signs remaining after decomposition.
    """
    N = U.shape[0]
    assert U.shape == (N, N), f"Expected square matrix, got {U.shape}"

    W = U.astype(np.float64).copy()
    thetas = []

    for c in range(N - 1):
        for r in range(N - 1, c, -1):
            a = W[r - 1, c]
            b = W[r, c]
            theta = math.atan2(b, a)

            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            row_top = W[r - 1, :].copy()
            row_bot = W[r, :].copy()
            W[r - 1, :] = cos_t * row_top + sin_t * row_bot
            W[r, :]     = -sin_t * row_top + cos_t * row_bot

            thetas.append((r - 1, r, theta))

    return thetas, np.diag(W).copy()


def reconstruct_unitary(thetas, diag):
    """Reconstruct orthogonal matrix from Givens rotations + diagonal.

    Inverse of decompose_unitary: applies rotations in reverse order.
    """
    N = len(diag)
    W = np.diag(diag.astype(np.float64))

    for (i, j, theta) in reversed(thetas):
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        row_i = W[i, :].copy()
        row_j = W[j, :].copy()
        W[i, :] = cos_t * row_i - sin_t * row_j
        W[j, :] = sin_t * row_i + cos_t * row_j

    return W


# =====================================================================
# Part 1 -- Phase quantization
# =====================================================================

def quantize_phases(phases, bits=4):
    """Quantize continuous phases to 2^bits discrete levels in [0, 2*pi).

    Ning 2024: MZI effective precision 4-6 bits.
    """
    phases = np.asarray(phases, dtype=np.float64)
    n_levels = 2 ** bits
    step = 2.0 * np.pi / n_levels
    wrapped = phases % (2.0 * np.pi)
    quantized = np.round(wrapped / step) * step
    return quantized % (2.0 * np.pi)


# =====================================================================
# Part 2 -- Hardware imperfection injection
# =====================================================================

def inject_phase_noise(thetas, sigma_phi=0.005, rng=None):
    """Add independent phase encoding noise to MZI angles.

    Shen 2017: "sigma_phi ~ 5 x 10^-3 radians on individual MZIs."
    Each phase shifter has a small random offset from its programmed
    value due to DAC imprecision and thermal fluctuation.

    Args:
        thetas: list of (row_i, row_j, theta) from decompose_unitary.
        sigma_phi: std of additive Gaussian noise per phase (radians).
                   Default 0.005 (Shen 2017 measurement).
        rng: numpy RandomState for reproducibility.

    Returns:
        list of (row_i, row_j, noisy_theta) -- same structure, perturbed.
    """
    if rng is None:
        rng = np.random.RandomState()
    return [
        (i, j, theta + rng.normal(0.0, sigma_phi))
        for (i, j, theta) in thetas
    ]


def inject_thermal_crosstalk(thetas, sigma_t=0.01, coupling=0.3, rng=None):
    """Add correlated thermal crosstalk noise to MZI phases.

    Ning 2024: "Adjacent thermo-optic phase shifters heat each other.
    Crosstalk coefficients decay with distance but are significant for
    nearest neighbors."

    Shen 2017: "Thermal crosstalk is the dominant excess noise."

    Model: generate independent noise per phase, then add a fraction
    (coupling) of each neighbour's noise.  This produces nearest-
    neighbour spatial correlation matching the physical heat diffusion.

    Args:
        thetas: list of (row_i, row_j, theta).
        sigma_t: base noise std.  Default 0.01.
        coupling: fraction of neighbour noise added.  Default 0.3.
                  0.0 = independent, 1.0 = fully correlated with neighbour.
        rng: numpy RandomState.

    Returns:
        list of (row_i, row_j, noisy_theta).
    """
    if rng is None:
        rng = np.random.RandomState()

    n = len(thetas)
    if n == 0:
        return thetas

    # Independent base noise.
    base = rng.normal(0.0, sigma_t, size=n)

    # Add coupled noise from nearest neighbours.
    correlated = base.copy()
    if n > 1:
        correlated[1:]  += coupling * base[:-1]   # left neighbour
        correlated[:-1] += coupling * base[1:]     # right neighbour

    return [
        (i, j, theta + correlated[k])
        for k, (i, j, theta) in enumerate(thetas)
    ]


def apply_optical_loss(sigma, n_stages, loss_db_per_stage=0.1):
    """Apply cumulative optical insertion loss to singular values.

    Ning 2024: "approximately 0.1 dB per MZI stage."
    Light amplitude attenuates as it passes through beamsplitters.
    Cumulative: total_dB = n_stages x loss_per_stage.
    Amplitude scales by 10^(-total_dB / 20).

    Args:
        sigma: (k,) singular values from SVD.
        n_stages: number of MZI stages (columns) light passes through.
        loss_db_per_stage: insertion loss per stage in dB.  Default 0.1.

    Returns:
        (k,) attenuated singular values.
    """
    total_db = n_stages * loss_db_per_stage
    amplitude_factor = 10.0 ** (-total_db / 20.0)
    return sigma * amplitude_factor


def apply_detector_noise(signal, sigma_s=0.02, rng=None):
    """Add photodetector shot noise to an output signal vector.

    Shen 2017: "sigma_D ~ 0.1%."
    Ning 2024: "shot noise sigma = 0.01 to 0.03."

    Applied at the OUTPUT of the matrix-vector multiply, not to the
    weight matrix itself.  For exp6 simulation, this is added after
    computing W_degraded @ x.

    Args:
        signal: numpy array (any shape) -- the output of W @ x.
        sigma_s: noise std.  Default 0.02.
        rng: numpy RandomState.

    Returns:
        signal + noise (same shape).
    """
    if rng is None:
        rng = np.random.RandomState()
    return signal + rng.normal(0.0, sigma_s, size=signal.shape)


# =====================================================================
# MZIProfiler -- full pipeline
# =====================================================================

class MZIProfiler:
    """Decompose and simulate a weight matrix on photonic hardware.

    Full pipeline (Part 1 + Part 2):
        1. SVD:  W = U @ diag(sigma) @ V^T
        2. Reck: U, V -> Givens MZI phases
        3. Quantize phases to `bits`-bit precision
        4. Inject phase encoding noise  (sigma_phi, Shen 2017)
        5. Inject thermal crosstalk     (sigma_t, correlated, Ning 2024)
        6. Reconstruct weight matrix from degraded phases
        7. Apply optical loss to singular values (0.1 dB/stage, Ning 2024)
        8. Detector noise applied separately at inference (sigma_s)

    Args:
        bits: phase quantization precision.  Default 4 (Ning 2024).
    """

    def __init__(self, bits: int = 4) -> None:
        self.bits = bits

    # ----- Part 1: decompose / quantize / reconstruct -----

    def decompose(self, W):
        """Decompose weight matrix into MZI phase parameters via SVD."""
        W = np.asarray(W, dtype=np.float64)
        M, N = W.shape

        U, sigma, Vt = np.linalg.svd(W, full_matrices=True)
        V = Vt.T

        U_thetas, U_diag = decompose_unitary(U)
        V_thetas, V_diag = decompose_unitary(V)

        n_mzis_U = M * (M - 1) // 2
        n_mzis_V = N * (N - 1) // 2

        return {
            "U_thetas": U_thetas,
            "U_diag": U_diag,
            "sigma": sigma,
            "V_thetas": V_thetas,
            "V_diag": V_diag,
            "n_mzis": n_mzis_U + n_mzis_V,
            "n_phases": 2 * (n_mzis_U + n_mzis_V) + M + N,
        }

    def quantize(self, decomposition):
        """Quantize all MZI phases to self.bits precision."""
        d = decomposition

        U_thetas_q = [
            (i, j, float(quantize_phases(np.array([theta]), self.bits)[0]))
            for (i, j, theta) in d["U_thetas"]
        ]
        V_thetas_q = [
            (i, j, float(quantize_phases(np.array([theta]), self.bits)[0]))
            for (i, j, theta) in d["V_thetas"]
        ]

        U_diag_phases = np.where(d["U_diag"] >= 0, 0.0, np.pi)
        V_diag_phases = np.where(d["V_diag"] >= 0, 0.0, np.pi)
        U_diag_q = np.where(
            quantize_phases(U_diag_phases, self.bits) < np.pi / 2,
            1.0, -1.0
        )
        V_diag_q = np.where(
            quantize_phases(V_diag_phases, self.bits) < np.pi / 2,
            1.0, -1.0
        )

        return {
            "U_thetas": U_thetas_q,
            "U_diag": U_diag_q,
            "sigma": d["sigma"].copy(),
            "V_thetas": V_thetas_q,
            "V_diag": V_diag_q,
            "n_mzis": d["n_mzis"],
            "n_phases": d["n_phases"],
            "quantization_bits": self.bits,
        }

    def reconstruct(self, decomposition):
        """Reconstruct weight matrix from MZI phases."""
        d = decomposition

        U_rec = reconstruct_unitary(d["U_thetas"], d["U_diag"])
        V_rec = reconstruct_unitary(d["V_thetas"], d["V_diag"])

        M = len(d["U_diag"])
        N = len(d["V_diag"])
        k = len(d["sigma"])

        S = np.zeros((M, N), dtype=np.float64)
        for i in range(k):
            S[i, i] = d["sigma"][i]

        return U_rec @ S @ V_rec.T

    def profile(self, W):
        """Part-1 pipeline: decompose -> quantize -> reconstruct -> error."""
        W = np.asarray(W, dtype=np.float64)

        decomp = self.decompose(W)
        quant = self.quantize(decomp)

        W_recon = self.reconstruct(decomp)
        W_quant = self.reconstruct(quant)

        w_norm = np.linalg.norm(W)
        err_svd = np.linalg.norm(W - W_recon)
        err_quant = np.linalg.norm(W - W_quant)

        return {
            "decomposition": decomp,
            "quantized": quant,
            "W_original": W,
            "W_reconstructed": W_recon,
            "W_quantized": W_quant,
            "error_svd": float(err_svd),
            "error_quantized": float(err_quant),
            "relative_error": float(err_quant / w_norm) if w_norm > 0 else 0.0,
        }

    # ----- Part 2: full hardware simulation -----

    def simulate(self, W, sigma_phi=0.005, sigma_t=0.01, sigma_s=0.02,
                 coupling=0.3, loss_db_per_stage=0.1, seed=None):
        """Full photonic hardware simulation pipeline.

        Chains all imperfections that a real chip imposes on a weight matrix:
            1. SVD decomposition to MZI phases
            2. Phase quantization (4-bit precision)
            3. Phase encoding noise (sigma_phi)
            4. Thermal crosstalk (sigma_t, correlated across neighbours)
            5. Reconstruct degraded weight matrix
            6. Optical insertion loss on singular values

        Detector noise (sigma_s) is NOT baked into the matrix -- it is
        returned as a convenience function for applying at inference time.

        Args:
            W: (M, N) weight matrix (one Monarch block).
            sigma_phi: phase encoding noise std (Shen 2017: 5e-3 rad).
            sigma_t:   thermal crosstalk noise std (Ning 2024: 0.01).
            sigma_s:   detector noise std (Ning 2024: 0.02).  Stored
                       for use in apply_detector_noise().
            coupling:  nearest-neighbour thermal coupling (0-1).
            loss_db_per_stage: optical loss per MZI column (Ning 2024: 0.1).
            seed: random seed for reproducibility.

        Returns:
            dict with keys:
                'W_original':       original weight matrix.
                'W_simulated':      degraded matrix (all imperfections).
                'W_quantized_only': degraded by quantization only (no noise).
                'error_total':      ||W - W_simulated||_F.
                'error_quant_only': ||W - W_quantized_only||_F.
                'relative_error':   error_total / ||W||_F.
                'optical_loss_db':  total optical loss in dB.
                'optical_loss_frac': fraction of power lost.
                'n_mzis':           total MZI count.
                'n_stages':         MZI stages (for loss computation).
                'sigma_s':          detector noise std (for inference).
        """
        rng = np.random.RandomState(seed)
        W = np.asarray(W, dtype=np.float64)
        M, N = W.shape

        # 1. SVD decomposition.
        decomp = self.decompose(W)

        # 2. Phase quantization.
        quant = self.quantize(decomp)
        W_quant_only = self.reconstruct(quant)

        # 3. Phase encoding noise (independent per phase).
        noisy_U = inject_phase_noise(quant["U_thetas"], sigma_phi, rng)
        noisy_V = inject_phase_noise(quant["V_thetas"], sigma_phi, rng)

        # 4. Thermal crosstalk (correlated across neighbours).
        noisy_U = inject_thermal_crosstalk(noisy_U, sigma_t, coupling, rng)
        noisy_V = inject_thermal_crosstalk(noisy_V, sigma_t, coupling, rng)

        # 5. Reconstruct from degraded phases.
        noisy_decomp = {
            "U_thetas": noisy_U,
            "U_diag": quant["U_diag"],
            "sigma": quant["sigma"].copy(),
            "V_thetas": noisy_V,
            "V_diag": quant["V_diag"],
        }

        # 6. Optical loss: attenuate singular values.
        #    n_stages = number of MZI columns for both U and V meshes.
        n_stages_U = M - 1   # triangular Reck mesh depth for MxM unitary
        n_stages_V = N - 1
        total_stages = n_stages_U + n_stages_V
        noisy_decomp["sigma"] = apply_optical_loss(
            noisy_decomp["sigma"], total_stages, loss_db_per_stage
        )

        W_sim = self.reconstruct(noisy_decomp)

        # Metrics.
        w_norm = np.linalg.norm(W)
        total_loss_db = total_stages * loss_db_per_stage
        loss_frac = 1.0 - 10.0 ** (-total_loss_db / 10.0)

        return {
            "W_original": W,
            "W_simulated": W_sim,
            "W_quantized_only": W_quant_only,
            "error_total": float(np.linalg.norm(W - W_sim)),
            "error_quant_only": float(np.linalg.norm(W - W_quant_only)),
            "relative_error": float(np.linalg.norm(W - W_sim) / w_norm)
                              if w_norm > 0 else 0.0,
            "optical_loss_db": float(total_loss_db),
            "optical_loss_frac": float(loss_frac),
            "n_mzis": decomp["n_mzis"],
            "n_stages": total_stages,
            "sigma_s": sigma_s,
        }


# ---------------------------------------------------------------------------
# Self-contained tests -- run with: python hardware/mzi_profiler.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    rng = np.random.RandomState(42)

    print("Testing hardware/mzi_profiler.py ...")
    print()

    # --- Test 1: Part 1 -- decompose -> reconstruct is exact ---
    N = 8
    Q, _ = np.linalg.qr(rng.randn(N, N))
    thetas, diag = decompose_unitary(Q)
    Q_recon = reconstruct_unitary(thetas, diag)
    err = np.linalg.norm(Q - Q_recon)
    assert err < 1e-10, f"Reconstruction error too large: {err}"
    assert len(thetas) == N * (N - 1) // 2
    print(f"  [PASS] Test 1 -- decompose/reconstruct {N}x{N} orthogonal:")
    print(f"         ||Q - Q_recon||_F = {err:.2e}")
    print(f"         MZI count = {len(thetas)} (expected {N*(N-1)//2})")

    # --- Test 2: Part 1 -- quantize_phases levels ---
    angles = rng.uniform(0, 2 * np.pi, size=1000)
    for bits in [4, 5, 6]:
        q = quantize_phases(angles, bits=bits)
        n_unique = len(np.unique(np.round(q, 10)))
        assert n_unique <= 2 ** bits
    print(f"  [PASS] Test 2 -- quantize_phases:")
    for bits in [4, 5, 6]:
        q = quantize_phases(angles, bits=bits)
        n_unique = len(np.unique(np.round(q, 10)))
        print(f"         {bits}-bit -> {n_unique} unique levels (<= {2**bits})")

    # --- Test 3: Part 2 -- noise injection increases error ---
    profiler = MZIProfiler(bits=4)
    W = rng.randn(8, 8)
    result_clean = profiler.profile(W)
    result_noisy = profiler.simulate(W, sigma_phi=0.005, sigma_t=0.01,
                                     loss_db_per_stage=0.1, seed=42)

    assert result_noisy["error_total"] > result_clean["error_quantized"], (
        "Hardware sim should produce MORE error than quantization alone"
    )
    assert result_noisy["error_total"] > 0
    assert result_noisy["optical_loss_db"] > 0
    assert 0 < result_noisy["optical_loss_frac"] < 1.0
    print(f"  [PASS] Test 3 -- simulate() adds error beyond quantization:")
    print(f"         quant-only error  = {result_clean['error_quantized']:.4f}")
    print(f"         full sim error    = {result_noisy['error_total']:.4f}")
    print(f"         optical loss      = {result_noisy['optical_loss_db']:.1f} dB "
          f"({result_noisy['optical_loss_frac']:.1%} power)")

    # --- Test 4: Part 2 -- detector noise is additive, correct std ---
    signal = np.ones(1000)
    noisy = apply_detector_noise(signal, sigma_s=0.02,
                                 rng=np.random.RandomState(0))
    noise = noisy - signal
    assert abs(np.std(noise) - 0.02) < 0.005, (
        f"Detector noise std = {np.std(noise):.4f}, expected ~0.02"
    )
    assert abs(np.mean(noise)) < 0.005, (
        f"Detector noise mean = {np.mean(noise):.4f}, expected ~0.0"
    )
    print(f"  [PASS] Test 4 -- detector noise (sigma_s=0.02):")
    print(f"         noise std  = {np.std(noise):.4f} (expected ~0.02)")
    print(f"         noise mean = {np.mean(noise):.4f} (expected ~0.00)")

    print()
    print("All tests passed.")
    print()
    print("Full pipeline: W -> SVD -> Reck -> quantize(4-bit)")
    print("  -> phase noise(5e-3) -> thermal crosstalk(0.01)")
    print("  -> reconstruct -> optical loss(0.1dB/stage)")
    print("  -> detector noise(0.02) at inference")
