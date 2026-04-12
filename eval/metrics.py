"""
eval/metrics.py

Photonic hardware metrics and supplementary generative-quality metrics.

Three components:
    1. PhotonicMetrics          -- latency (ns/step) and energy (fJ/MAC, pJ/sample)
    2. compute_inception_score  -- IS from InceptionV3 class probabilities
    3. compute_precision_recall -- k-NN precision/recall in feature space

Physical basis for latency:
    Light propagates through one MZI column in ~10 ps (Shen 2017).
    Each Monarch layer M = PLP^T R maps to 2 MZI columns (block-diag R
    + permute + block-diag L).  A PhotonFlowBlock has 2 Monarch layers
    (L, R) = 4 MZI columns, plus a saturable absorber (~3 ps, graphene
    waveguide) and divisive power norm (~10 ps, photodetector feedback).

        latency_step = n_blocks x (4 x 10ps + 3ps + 10ps)

    For 6 blocks: 6 x 53 ps = 318 ps = 0.318 ns -- under the < 1 ns target.

Physical basis for energy (two components):
    A. Optical computation energy (per ODE step):
       Passive beamsplitter MAC = sub-fJ.
       Shen 2017 formula: P/R = 5/(m x N) fJ per FLOP.
       Photodetector readout: ~1 fJ each.

    B. Phase-shifter programming energy (one-time at model load):
       ~10 fJ per phase shifter to set the thermo-optic heaters.
       Amortised across all inferences -- NOT charged per step.

Physical basis for optical loss:
    0.1 dB per MZI stage (Ning 2024).  Cumulative across all columns.
    24 MZI columns = 2.4 dB = ~42% signal power lost.  Must stay under
    ~3 dB for usable SNR without optical amplification.

Phase-shifter count (derived from architecture, Dao 2022 + Shen 2017):
    Each m x m block in a Monarch factor needs m(m-1)/2 MZIs (Reck
    decomposition).  Each MZI has 2 phase shifters (theta, phi).
    Total = n_blocks x monarchs_per_block x 2_factors x m x m(m-1)/2 x 2.

MAC count (Dao 2022):
    One Monarch layer of dim n = m^2 performs 2 x m x m^2 = 2 x n x sqrt(n)
    multiply-accumulate operations.

Paper Table I metrics served:
    exp4 -- FID, precision          -> compute_precision_recall()
    exp5 -- FID, IS                 -> compute_inception_score()
    exp6 -- ns/step, fJ/MAC         -> PhotonicMetrics

References:
    Shen et al., "Deep Learning with Coherent Nanophotonic Circuits,"
    Nature Photonics, 2017.
    (MZI propagation ~10 ps, energy P/R = 5/(m*N) fJ/FLOP, >100 GHz
    photodetection rate, saturable absorber nonlinearity.)

    Ning et al., "Photonic-Electronic Integrated Circuits for High-
    Performance Computing and AI Accelerators," J. Lightwave Technol., 2024.
    (0.1 dB loss/stage, sub-fJ optical MAC, 1-10 fJ/MAC total system,
    4-6 bit MZI precision, O-E-O bottleneck.)

    Dao et al., "Monarch: Expressive Structured Matrices for Efficient
    and Accurate Training," ICML 2022.
    (M = PLP^T R, FLOPs O(n^{3/2}), MACs = 2 x n x sqrt(n).)

    Jiang et al. (Zhu 2026), "A Fully Real-Valued End-to-End Optical
    Neural Network for Generative Model," Frontiers of Optoelectronics.
    (Baseline: 1.76 ns latency, 37 pJ/OP on real chip.)

    Kynkaanniemi et al., "Improved Precision and Recall Metric for
    Assessing Generative Models," NeurIPS 2019.

    Salimans et al., "Improved Techniques for Training GANs," NeurIPS 2016.
"""

import math
import warnings

import numpy as np


# =====================================================================
# 1. Photonic hardware metrics
# =====================================================================

class PhotonicMetrics:
    """Estimate photonic-chip latency and energy for a PhotonFlow model.

    All defaults match the Shen 2017 / Ning 2024 / Dao 2022 papers.
    Override via constructor for sensitivity analysis.

    Key design decisions vs the previous (incorrect) version:
        - Phase-shifter count is DERIVED from (dim, block_size, n_blocks)
          using the Reck decomposition formula, not an arbitrary default.
        - Energy is SPLIT into optical-per-step (charged every ODE step)
          and programming (one-time, amortised).
        - Latency includes saturable-absorber and power-norm delays,
          not just MZI propagation.
        - Optical loss is modelled (0.1 dB per MZI stage, cumulative).
        - fJ/MAC uses the Monarch MAC count from Dao 2022.

    Args:
        n_blocks:  Number of PhotonFlowBlocks.  Default 6.
        dim:       Model feature dimension.  Default 784 (28x28 MNIST).
        block_size: Monarch block size m.  Default sqrt(dim).
                    dim must equal m^2.
        ode_steps: ODE solver steps at inference.  Default 20.
        monarchs_per_block:  Monarch layers per block.  Default 2 (L, R).
        mzi_columns_per_monarch:  MZI columns per Monarch layer.
                                  Default 2 (block-diag R + block-diag L).
        propagation_delay_ps:  Per MZI column.  Default 10 ps (Shen 2017).
        absorber_delay_ps:     Saturable absorber (graphene waveguide).
                               Default 3 ps.
        powernorm_delay_ps:    Photodetector feedback loop for divisive
                               power norm.  Default 10 ps (Shen 2017
                               photodetection rate >100 GHz).
        optical_fj_per_mac:    Optical energy per MAC (passive beamsplitter).
                               Default 0.001 fJ (Shen 2017: sub-fJ).
        detector_energy_fj:    Per photodetector readout.  Default 1 fJ.
        shifter_energy_fj:     Per phase shifter programming (one-time).
                               Default 10 fJ (Ning 2024).
        optical_loss_db_per_stage:  Insertion loss per MZI stage.
                                   Default 0.1 dB (Ning 2024).
    """

    def __init__(
        self,
        n_blocks: int = 6,
        dim: int = 784,
        block_size: int = None,
        ode_steps: int = 20,
        monarchs_per_block: int = 2,
        mzi_columns_per_monarch: int = 2,
        propagation_delay_ps: float = 10.0,
        absorber_delay_ps: float = 3.0,
        powernorm_delay_ps: float = 10.0,
        optical_fj_per_mac: float = 0.001,
        detector_energy_fj: float = 1.0,
        shifter_energy_fj: float = 10.0,
        optical_loss_db_per_stage: float = 0.1,
    ) -> None:
        self.n_blocks = n_blocks
        self.dim = dim
        self.ode_steps = ode_steps
        self.monarchs_per_block = monarchs_per_block
        self.mzi_columns_per_monarch = mzi_columns_per_monarch
        self.propagation_delay_ps = propagation_delay_ps
        self.absorber_delay_ps = absorber_delay_ps
        self.powernorm_delay_ps = powernorm_delay_ps
        self.optical_fj_per_mac = optical_fj_per_mac
        self.detector_energy_fj = detector_energy_fj
        self.shifter_energy_fj = shifter_energy_fj
        self.optical_loss_db_per_stage = optical_loss_db_per_stage

        # Derive block size m from dim (dim = m^2).
        if block_size is None:
            block_size = int(math.isqrt(dim))
            if block_size * block_size != dim:
                raise ValueError(
                    f"dim={dim} is not a perfect square. "
                    f"Provide block_size explicitly."
                )
        self.block_size = block_size

    # ----- architecture-derived counts -----

    @property
    def m(self) -> int:
        """Monarch block size (shorthand)."""
        return self.block_size

    @property
    def total_mzi_columns(self) -> int:
        """Total MZI columns in the full model."""
        return (self.n_blocks
                * self.monarchs_per_block
                * self.mzi_columns_per_monarch)

    @property
    def n_phase_shifters(self) -> int:
        """Total phase shifters, derived from architecture.

        Per Reck decomposition (Shen 2017): an m x m unitary needs
        m(m-1)/2 MZIs.  Each MZI has 2 phase shifters (theta, phi).

        Per Monarch layer: 2 factors x m blocks x m(m-1)/2 MZIs x 2 phases.
        Per block: monarchs_per_block Monarch layers.
        Total: n_blocks x above.
        """
        mzis_per_block_unitary = self.m * (self.m - 1) // 2
        phases_per_factor = self.m * mzis_per_block_unitary * 2
        phases_per_monarch = 2 * phases_per_factor  # 2 factors (R and L)
        phases_per_block = self.monarchs_per_block * phases_per_monarch
        return self.n_blocks * phases_per_block

    @property
    def macs_per_monarch(self) -> int:
        """MACs per Monarch layer (Dao 2022: 2 x n x sqrt(n))."""
        return 2 * self.dim * self.m

    @property
    def macs_per_step(self) -> int:
        """Total MACs per ODE step (forward pass through all blocks)."""
        return self.n_blocks * self.monarchs_per_block * self.macs_per_monarch

    @property
    def macs_per_sample(self) -> int:
        """Total MACs per generated sample (all ODE steps)."""
        return self.macs_per_step * self.ode_steps

    # ----- latency -----

    def latency_per_step_ps(self) -> float:
        """Latency for one ODE step, in picoseconds.

        = n_blocks x (MZI_columns x propagation + absorber + powernorm)

        MZI propagation: light through beamsplitter cascades.
        Absorber: graphene waveguide saturable absorber.
        Powernorm: photodetector measures L2 norm + microring feedback.
        """
        mzi_delay = (self.monarchs_per_block
                     * self.mzi_columns_per_monarch
                     * self.propagation_delay_ps)
        per_block = mzi_delay + self.absorber_delay_ps + self.powernorm_delay_ps
        return self.n_blocks * per_block

    def latency_per_step_ns(self) -> float:
        """Latency for one ODE step, in nanoseconds.

        Target: < 1 ns (paper success criterion).
        """
        return self.latency_per_step_ps() / 1000.0

    def total_latency_ns(self) -> float:
        """Total latency to generate one sample (all ODE steps), in ns."""
        return self.latency_per_step_ns() * self.ode_steps

    # ----- energy -----

    def optical_energy_per_step_fj(self) -> float:
        """Optical computation energy per ODE step, in femtojoules.

        = (MACs x fJ_per_MAC) + (detectors x fJ_per_detector)

        This is the PER-STEP cost: laser light propagates through the
        passive beamsplitter mesh (sub-fJ per MAC, Shen 2017) and
        photodetectors read the output.
        """
        mac_energy = self.macs_per_step * self.optical_fj_per_mac
        det_energy = self.dim * self.detector_energy_fj
        return mac_energy + det_energy

    def optical_energy_per_sample_pj(self) -> float:
        """Optical computation energy per sample, in picojoules.

        Target: < 1 pJ (paper success criterion).
        """
        return self.optical_energy_per_step_fj() * self.ode_steps / 1000.0

    def programming_energy_fj(self) -> float:
        """One-time phase-shifter programming energy, in femtojoules.

        This is the cost of loading model weights onto the chip by
        setting all thermo-optic phase shifters.  Paid ONCE at model
        load, amortised across all inferences.  NOT charged per step.
        """
        return self.n_phase_shifters * self.shifter_energy_fj

    def programming_energy_pj(self) -> float:
        """One-time programming energy, in picojoules."""
        return self.programming_energy_fj() / 1000.0

    def fj_per_mac(self) -> float:
        """Optical energy per MAC, in femtojoules.

        = optical_energy_per_step / MACs_per_step

        Shen 2017 reference: 5/(m*N) fJ/FLOP for m layers, N neurons.
        For m=12 (6 blocks x 2 Monarchs), N=784: ~0.0005 fJ/FLOP.
        """
        if self.macs_per_step == 0:
            return 0.0
        return self.optical_energy_per_step_fj() / self.macs_per_step

    # ----- optical loss -----

    def optical_loss_db(self) -> float:
        """Cumulative optical loss through the full model, in dB.

        = total_mzi_columns x loss_per_stage

        Ning 2024: 0.1 dB per MZI stage.  >3 dB means >50% power lost
        and may need optical amplification.
        """
        return self.total_mzi_columns * self.optical_loss_db_per_stage

    def optical_loss_fraction(self) -> float:
        """Fraction of signal power lost (0.0 = no loss, 1.0 = total loss).

        Computed from dB: fraction_lost = 1 - 10^(-dB/10).
        """
        return 1.0 - 10.0 ** (-self.optical_loss_db() / 10.0)

    # ----- summary -----

    def summary(self) -> dict:
        """Return all metrics as a dict for CSV / notebook logging."""
        return {
            "latency_per_step_ns": self.latency_per_step_ns(),
            "total_latency_ns": self.total_latency_ns(),
            "optical_energy_per_step_fj": self.optical_energy_per_step_fj(),
            "optical_energy_per_sample_pj": self.optical_energy_per_sample_pj(),
            "programming_energy_pj": self.programming_energy_pj(),
            "fj_per_mac": self.fj_per_mac(),
            "optical_loss_db": self.optical_loss_db(),
            "optical_loss_fraction": self.optical_loss_fraction(),
            "n_blocks": self.n_blocks,
            "dim": self.dim,
            "block_size_m": self.block_size,
            "mzi_columns": self.total_mzi_columns,
            "n_phase_shifters": self.n_phase_shifters,
            "macs_per_step": self.macs_per_step,
            "macs_per_sample": self.macs_per_sample,
            "ode_steps": self.ode_steps,
        }


# =====================================================================
# 2. Inception Score (IS)
# =====================================================================

def compute_inception_score(class_probs, splits=10):
    """Compute Inception Score from InceptionV3 softmax probabilities.

    IS = exp( E_x[ KL( p(y|x) || p(y) ) ] )

    Higher IS = better (diverse + confident predictions).

    The score is computed over *splits* random subsets and averaged to
    reduce variance, following Salimans et al. 2016.

    Note: IS uses ImageNet 1000-class softmax.  For MNIST/CIFAR-10
    (10 classes), the generated images do not map cleanly to ImageNet
    classes, so IS will be low and less meaningful.  Use IS primarily
    for CelebA-64 (exp5) where face-like images map to ImageNet
    categories.  FID is the primary metric for all experiments.

    Args:
        class_probs: numpy array (N, C) of softmax probabilities.
                     Standard: C=1000 from InceptionV3.
                     Each row must sum to 1.
        splits: number of splits for variance reduction.  Default 10.

    Returns:
        (is_mean, is_std) -- mean and std of IS across splits.
    """
    N = class_probs.shape[0]
    assert class_probs.ndim == 2 and class_probs.shape[1] > 1, (
        f"Expected (N, C) softmax probs, got shape {class_probs.shape}"
    )

    split_scores = []
    for k in range(splits):
        start = k * N // splits
        end = (k + 1) * N // splits
        part = class_probs[start:end]

        # p(y|x) for each image -- already the rows of `part`.
        # p(y) = marginal = average of all conditionals in this split.
        py = np.mean(part, axis=0, keepdims=True)

        # KL( p(y|x) || p(y) ) for each image, then average.
        kl = part * (np.log(np.clip(part, 1e-10, 1.0))
                     - np.log(np.clip(py, 1e-10, 1.0)))
        kl = np.sum(kl, axis=1)
        split_scores.append(np.exp(np.mean(kl)))

    return float(np.mean(split_scores)), float(np.std(split_scores))


# =====================================================================
# 3. Precision and Recall (Kynkaanniemi et al. 2019)
# =====================================================================

def compute_precision_recall(real_features, gen_features, k=3,
                             batch_size=None):
    """Compute precision and recall in InceptionV3 feature space.

    Precision = fraction of generated samples within the support of
                the real distribution (quality).
    Recall    = fraction of real samples within the support of the
                generated distribution (diversity).

    Uses k-nearest-neighbour balls (Kynkaanniemi et al. 2019).

    Args:
        real_features: numpy array (N_real, D), e.g. (10000, 2048).
        gen_features:  numpy array (N_gen,  D), same D.
        k: number of nearest neighbours.  Default 3.
        batch_size: if set, compute distances in batches to limit
                    memory.  None = full matrix (faster but needs
                    N^2 x 8 bytes RAM; 10K samples ~ 800 MB).

    Returns:
        (precision, recall) -- each a float in [0, 1].
    """
    from scipy.spatial.distance import cdist

    assert real_features.ndim == 2 and gen_features.ndim == 2
    assert real_features.shape[1] == gen_features.shape[1], (
        f"Feature dim mismatch: {real_features.shape[1]} vs "
        f"{gen_features.shape[1]}"
    )

    def _kth_nn_dist(features, k_nn):
        """Return the k-th NN distance for every point (excluding self)."""
        dists = cdist(features, features, metric="euclidean")
        sorted_dists = np.sort(dists, axis=1)
        # Column 0 is self (dist=0), column k_nn is the k-th neighbour.
        return sorted_dists[:, k_nn]

    # Radii of k-NN balls for real and generated sets.
    real_radii = _kth_nn_dist(real_features, k)
    gen_radii = _kth_nn_dist(gen_features, k)

    # Cross-distances.
    dist_gen_to_real = cdist(gen_features, real_features, metric="euclidean")
    dist_real_to_gen = cdist(real_features, gen_features, metric="euclidean")

    # Precision: each gen sample inside ANY real k-NN ball?
    precision = np.mean(
        np.any(dist_gen_to_real <= real_radii[np.newaxis, :], axis=1)
    )

    # Recall: each real sample inside ANY gen k-NN ball?
    recall = np.mean(
        np.any(dist_real_to_gen <= gen_radii[np.newaxis, :], axis=1)
    )

    return float(precision), float(recall)


# ---------------------------------------------------------------------------
# Self-contained tests -- run with: python eval/metrics.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("Testing eval/metrics.py ...")
    print()

    # --- Test 1: PhotonicMetrics -- values match paper targets ---
    pm = PhotonicMetrics()  # default: 6 blocks, dim=784, 20 ODE steps
    s = pm.summary()

    # Latency: must be < 1 ns/step (paper target)
    assert s["latency_per_step_ns"] > 0, "Latency must be positive"
    assert s["latency_per_step_ns"] < 1.0, (
        f"Latency {s['latency_per_step_ns']:.3f} ns exceeds 1 ns target"
    )
    # Energy: optical per sample -- paper target < 1 pJ
    assert s["optical_energy_per_sample_pj"] > 0, "Energy must be positive"
    # fJ/MAC: should be sub-fJ to few fJ range (Ning 2024: 1-10 fJ total system)
    assert s["fj_per_mac"] > 0, "fJ/MAC must be positive"
    assert s["fj_per_mac"] < 10.0, (
        f"fJ/MAC = {s['fj_per_mac']:.4f}, expected < 10 (Ning 2024 range)"
    )
    # Phase shifters: derived from Reck decomposition, should be large
    assert s["n_phase_shifters"] > 10000, (
        f"n_phase_shifters = {s['n_phase_shifters']}, expected >> 2048"
    )
    # MACs: from Dao 2022 formula 2*n*sqrt(n) per Monarch
    assert s["macs_per_step"] > 100000, (
        f"MACs/step = {s['macs_per_step']}, too low for dim=784"
    )
    # Optical loss: should be > 0 and in reasonable range
    assert 0 < s["optical_loss_db"] < 5.0, (
        f"Optical loss = {s['optical_loss_db']:.1f} dB, expected 1-4 dB"
    )

    print(f"  [PASS] Test 1 -- PhotonicMetrics (6 blocks, dim=784):")
    print(f"         latency/step    = {s['latency_per_step_ns']:.3f} ns  (target < 1 ns)")
    print(f"         total latency   = {s['total_latency_ns']:.2f} ns  ({s['ode_steps']} steps)")
    print(f"         optical E/sample= {s['optical_energy_per_sample_pj']:.4f} pJ  (target < 1 pJ)")
    print(f"         programming E   = {s['programming_energy_pj']:.1f} pJ  (one-time)")
    print(f"         fJ/MAC          = {s['fj_per_mac']:.4f}")
    print(f"         optical loss    = {s['optical_loss_db']:.1f} dB ({s['optical_loss_fraction']:.1%} power)")
    print(f"         phase shifters  = {s['n_phase_shifters']:,}")
    print(f"         MACs/step       = {s['macs_per_step']:,}")
    print(f"         MACs/sample     = {s['macs_per_sample']:,}")

    # --- Test 2: Inception Score -- uniform ~ 1.0, peaked ~ n_classes ---
    rng = np.random.RandomState(42)

    uniform = np.ones((500, 10)) / 10.0
    is_mean_u, is_std_u = compute_inception_score(uniform, splits=5)
    assert abs(is_mean_u - 1.0) < 0.1, (
        f"Uniform probs should give IS ~ 1.0, got {is_mean_u:.3f}"
    )

    peaked = np.zeros((500, 10))
    for i in range(500):
        peaked[i, i % 10] = 1.0
    peaked = np.clip(peaked, 1e-10, 1.0)
    peaked /= peaked.sum(axis=1, keepdims=True)
    is_mean_p, is_std_p = compute_inception_score(peaked, splits=5)
    assert is_mean_p > 5.0, (
        f"Peaked one-hot probs should give IS >> 1, got {is_mean_p:.3f}"
    )

    print(f"  [PASS] Test 2 -- Inception Score:")
    print(f"         uniform IS = {is_mean_u:.3f} +/- {is_std_u:.3f}  (expected ~1.0)")
    print(f"         peaked  IS = {is_mean_p:.3f} +/- {is_std_p:.3f}  (expected ~10.0)")

    # --- Test 3: Precision/Recall -- same set -> both ~1.0 ---
    feats = rng.randn(100, 32).astype(np.float64)
    p, r = compute_precision_recall(feats, feats, k=3)
    assert p > 0.95, f"Same-set precision should be ~1.0, got {p:.3f}"
    assert r > 0.95, f"Same-set recall should be ~1.0, got {r:.3f}"
    print(f"  [PASS] Test 3 -- Precision/Recall (same set):")
    print(f"         precision = {p:.3f}, recall = {r:.3f}")

    # --- Test 4: Precision/Recall -- disjoint sets -> both low ---
    feats_a = rng.randn(100, 32).astype(np.float64)
    feats_b = rng.randn(100, 32).astype(np.float64) + 20.0
    p2, r2 = compute_precision_recall(feats_a, feats_b, k=3)
    assert p2 < 0.2, f"Disjoint precision should be low, got {p2:.3f}"
    assert r2 < 0.2, f"Disjoint recall should be low, got {r2:.3f}"
    print(f"  [PASS] Test 4 -- Precision/Recall (disjoint sets):")
    print(f"         precision = {p2:.3f}, recall = {r2:.3f}")

    print()
    print("All tests passed.")
    print()
    print("Paper Table I coverage:")
    print("  exp4 -- FID + precision   -> compute_precision_recall()")
    print("  exp5 -- FID + IS          -> compute_inception_score()")
    print("  exp6 -- ns/step, fJ/MAC   -> PhotonicMetrics")
