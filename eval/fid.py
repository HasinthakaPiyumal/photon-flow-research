"""
eval/fid.py

Frechet Inception Distance (FID) calculator.

What is FID?
    FID measures the quality of generated images by comparing them to real
    images in the feature space of a pretrained InceptionV3 network.
    Lower FID = generated images are closer to real images = better model.

    The metric fits a multivariate Gaussian to the 2048-dim pool3 features
    of each set (real and generated), then computes the Frechet distance
    between the two Gaussians:

        FID = ‖μ₁ − μ₂‖² + Tr(Σ₁ + Σ₂ − 2(Σ₁ Σ₂)^½)

    Identical distributions → FID = 0.  Completely different → FID >> 0.

Why InceptionV3 pool3?
    The "pool3" layer is the 2048-dim output of the global average pool
    just before the final classifier.  These features capture high-level
    semantic content (shapes, textures, objects) without being tied to
    ImageNet class labels.  This is the standard layer used by the
    original FID paper and by pytorch-fid.

PhotonFlow usage:
    Every experiment in Table I of the paper requires FID:
        exp1 — baseline FID reference (GPU CFM + attention)
        exp2 — PhotonFlow FID delta vs exp1
        exp3 — FID with noise regularization
        exp4 — FID + precision after 4-bit QAT
        exp6 — FID after hardware simulation (MZI profiling)

    Success criterion: FID within 10% of the GPU attention baseline.

Dataset handling:
    MNIST   (1-ch, 28×28)  → grayscale repeated to 3-ch, resized to 299×299
    CIFAR-10 (3-ch, 32×32) → resized to 299×299
    CelebA-64 (3-ch, 64×64) → resized to 299×299
    All images expected in [0, 1] range, normalised with ImageNet stats.

References:
    Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge
    to a Local Nash Equilibrium," NeurIPS 2017.
    (Original FID definition — the Frechet distance formula we implement.)

    Peebles & Xie, "Scalable Diffusion Models with Transformers," ICCV 2023.
    (DiT baseline achieves FID 2.27 on ImageNet — our GPU comparison target.)

    spec: InceptionV3 pool3 features (2048-dim), Frechet distance formula.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from scipy import linalg


class InceptionV3Features(nn.Module):
    """InceptionV3 truncated at pool3 to output 2048-dim features.

    Built as a sequential chain of InceptionV3 sub-modules so we bypass
    the original forward() method (which has aux-logits branching and
    training-mode special cases we don't need).

    The chain runs: Conv2d_1a → ... → Mixed_7c → AdaptiveAvgPool → Flatten.
    Output: (batch, 2048) — the "pool3" features used for FID.

    All parameters are frozen (requires_grad=False) because we only use
    this network as a fixed feature extractor, never for training.
    """

    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights="DEFAULT", transform_input=False)

        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Extract 2048-dim pool3 features.

        Args:
            x: (B, 3, 299, 299) tensor, ImageNet-normalised.

        Returns:
            (B, 2048) feature tensor.
        """
        return self.blocks(x)


class FIDCalculator:
    """Frechet Inception Distance calculator.

    Provides the full pipeline:
        images → preprocess → InceptionV3 features → statistics → FID score

    Handles MNIST (1-ch 28×28), CIFAR-10 (3-ch 32×32), and CelebA-64
    (3-ch 64×64) by resizing to 299×299 and converting grayscale to RGB.

    Args:
        device: 'cuda', 'cpu', or None (auto-detect).
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = InceptionV3Features().to(self.device).eval()

    # ----- preprocessing -----

    def _preprocess(self, images):
        """Resize, convert grayscale, and normalise a batch for InceptionV3.

        Args:
            images: (B, C, H, W) tensor in [0, 1]. C may be 1 or 3.

        Returns:
            (B, 3, 299, 299) tensor normalised with ImageNet stats.
        """
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        images = F.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )

        mean = torch.tensor(self._MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(self._STD, device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std

    # ----- feature extraction -----

    @torch.no_grad()
    def extract_features(self, images, batch_size=64):
        """Extract 2048-dim pool3 features from images.

        Args:
            images: (N, C, H, W) tensor in [0, 1], or a DataLoader
                    yielding (images, labels) tuples.
            batch_size: batch size when *images* is a tensor (ignored
                        for DataLoader input).

        Returns:
            numpy array of shape (N, 2048).
        """
        parts = []

        if isinstance(images, torch.utils.data.DataLoader):
            for batch in images:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = self._preprocess(batch.to(self.device))
                parts.append(self.model(batch).cpu().numpy())
        else:
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                batch = self._preprocess(batch.to(self.device))
                parts.append(self.model(batch).cpu().numpy())

        return np.concatenate(parts, axis=0)

    # ----- statistics -----

    def compute_statistics(self, features):
        """Compute mean and covariance of feature vectors.

        Args:
            features: numpy array (N, D).

        Returns:
            (mu, sigma) — mu is (D,), sigma is (D, D).

        Raises:
            ValueError: if fewer than 2 samples are provided.
        """
        if features.shape[0] < 2:
            raise ValueError(
                f"Need >= 2 samples for covariance, got {features.shape[0]}"
            )
        if features.shape[0] < features.shape[1]:
            warnings.warn(
                f"Fewer samples ({features.shape[0]}) than feature dims "
                f"({features.shape[1]}). Covariance will be singular — "
                f"FID may be inaccurate. Use >= 10 000 samples for "
                f"reliable results.",
                stacklevel=2,
            )
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    # ----- Frechet distance -----

    @staticmethod
    def _stable_sqrtm(matrix):
        """Matrix square root with numerical-stability fallback.

        scipy.linalg.sqrtm can return complex values on near-singular
        matrices.  If the imaginary residual is tiny we discard it;
        otherwise we fall back to eigenvalue decomposition.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sqrtm_result, _ = linalg.sqrtm(matrix, disp=False)

        if np.iscomplexobj(sqrtm_result):
            if np.max(np.abs(sqrtm_result.imag)) <= 1e-3:
                sqrtm_result = sqrtm_result.real
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(matrix)
                eigenvalues = np.maximum(eigenvalues, 0.0)
                sqrtm_result = (
                    eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]
                ) @ eigenvectors.T

        return sqrtm_result

    def compute_fid(self, stats1, stats2, eps=1e-6):
        """Compute Frechet Inception Distance between two distributions.

        FID = ‖μ₁ − μ₂‖² + Tr(Σ₁ + Σ₂ − 2(Σ₁ Σ₂)^½)

        Args:
            stats1: (mu1, sigma1) from compute_statistics.
            stats2: (mu2, sigma2) from compute_statistics.
            eps: small constant added to the diagonal of the covariance
                 product before sqrtm, for numerical safety.

        Returns:
            FID score (float, >= 0).  Lower is better.
        """
        mu1, sigma1 = stats1
        mu2, sigma2 = stats2

        mu1 = np.atleast_1d(mu1).astype(np.float64)
        mu2 = np.atleast_1d(mu2).astype(np.float64)
        sigma1 = np.atleast_2d(sigma1).astype(np.float64)
        sigma2 = np.atleast_2d(sigma2).astype(np.float64)

        assert mu1.shape == mu2.shape, (
            f"Mean shape mismatch: {mu1.shape} vs {mu2.shape}"
        )
        assert sigma1.shape == sigma2.shape, (
            f"Covariance shape mismatch: {sigma1.shape} vs {sigma2.shape}"
        )

        # ‖μ₁ − μ₂‖²
        diff = mu1 - mu2
        mean_term = diff @ diff

        # (Σ₁ Σ₂)^½  — epsilon on diagonal for numerical safety.
        cov_product = sigma1 @ sigma2
        cov_product += eps * np.eye(cov_product.shape[0])
        cov_sqrt = self._stable_sqrtm(cov_product)

        # Tr(Σ₁ + Σ₂ − 2 (Σ₁ Σ₂)^½)
        trace_term = np.trace(sigma1 + sigma2 - 2.0 * cov_sqrt)

        return max(float(mean_term + trace_term), 0.0)

    # ----- convenience -----

    def compute_fid_from_images(self, real_images, generated_images,
                                batch_size=64):
        """End-to-end FID: images in → score out.

        Args:
            real_images: (N, C, H, W) tensor in [0, 1] or DataLoader.
            generated_images: same format as *real_images*.
            batch_size: batch size for feature extraction.

        Returns:
            FID score (float).
        """
        feats_real = self.extract_features(real_images, batch_size)
        feats_gen = self.extract_features(generated_images, batch_size)

        stats_real = self.compute_statistics(feats_real)
        stats_gen = self.compute_statistics(feats_gen)

        return self.compute_fid(stats_real, stats_gen)


# ---------------------------------------------------------------------------
# Self-contained tests — run with: python eval/fid.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # We test the Frechet distance math without InceptionV3 (no GPU needed).
    # InceptionV3-dependent tests run on Colab via the notebooks.

    rng = np.random.RandomState(42)

    # Build a lightweight calculator that skips InceptionV3 loading.
    class _MathOnly(FIDCalculator):
        def __init__(self):
            self.device = None
            self.model = None

    calc = _MathOnly()

    print("Testing FIDCalculator (Frechet distance math)...")
    print()

    # --- Test 1: same-vs-same FID ≈ 0 ---
    feats = rng.randn(500, 64).astype(np.float64)
    stats = calc.compute_statistics(feats)
    fid = calc.compute_fid(stats, stats)
    assert fid >= 0.0, f"FID should be non-negative, got {fid}"
    assert fid < 1.0, f"Same-vs-same FID should be ≈ 0, got {fid}"
    print(f"  [PASS] Test 1 — same-vs-same: FID = {fid:.8f}")

    # --- Test 2: shifted mean → FID high (> 50) ---
    feats_a = rng.randn(500, 64).astype(np.float64)
    feats_b = rng.randn(500, 64).astype(np.float64) + 5.0
    stats_a = calc.compute_statistics(feats_a)
    stats_b = calc.compute_statistics(feats_b)
    fid2 = calc.compute_fid(stats_a, stats_b)
    assert fid2 > 50.0, f"Different-distribution FID should be > 50, got {fid2}"
    print(f"  [PASS] Test 2 — shifted mean: FID = {fid2:.2f}")

    # --- Test 3: symmetry — FID(A,B) ≈ FID(B,A) ---
    fid_ab = calc.compute_fid(stats_a, stats_b)
    fid_ba = calc.compute_fid(stats_b, stats_a)
    assert abs(fid_ab - fid_ba) < 1.0, (
        f"FID should be symmetric: {fid_ab:.2f} vs {fid_ba:.2f}"
    )
    print(f"  [PASS] Test 3 — symmetry: FID(A,B)={fid_ab:.2f}, FID(B,A)={fid_ba:.2f}")

    # --- Test 4: near-singular covariance (N < D) — no crash, no NaN ---
    feats_small_a = rng.randn(10, 64).astype(np.float64)
    feats_small_b = rng.randn(10, 64).astype(np.float64) + 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress expected singular warning
        stats_sa = calc.compute_statistics(feats_small_a)
        stats_sb = calc.compute_statistics(feats_small_b)
    fid4 = calc.compute_fid(stats_sa, stats_sb)
    assert fid4 >= 0.0, f"FID should be >= 0, got {fid4}"
    assert not np.isnan(fid4), "FID is NaN on near-singular covariance"
    print(f"  [PASS] Test 4 — near-singular (N<D): FID = {fid4:.2f}, no NaN/crash")

    print()
    print("All tests passed.")
    print()
    print("FID formula: ‖μ₁ − μ₂‖² + Tr(Σ₁ + Σ₂ − 2(Σ₁ Σ₂)^½)")
    print("  same distribution → FID ≈ 0")
    print("  different distribution → FID >> 0")
    print("  PhotonFlow target: within 10% of GPU attention baseline FID")
