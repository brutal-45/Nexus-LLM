"""
Nexus Probability Distributions
=====================================
All common probability distributions implemented from scratch.

Distributions:
    - Gaussian (Normal): N(mu, sigma^2)
    - Categorical: Cat(p1, ..., pK)
    - Multinomial: Multi(n, p1, ..., pK)
    - Bernoulli: Ber(p)
    - Beta: Beta(alpha, beta)
    - Dirichlet: Dir(alpha1, ..., alphaK)
    - Uniform: U(a, b)
    - Mixture of Gaussians: sum_k pi_k * N(mu_k, sigma_k^2)
    - Empirical: distribution from data samples

Each distribution supports:
    - pdf/pmf: probability density/mass function
    - log_pdf/log_pmf: numerically stable log probability
    - sample: random sampling
    - mean, var, entropy: distribution properties
    - kl_divergence: KL divergence to another distribution of same family

All computations use numerically stable formulations to avoid
overflow/underflow (log-sum-exp trick, etc.).
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod

Array = np.ndarray


class Distribution(ABC):
    """Abstract base class for all probability distributions."""

    @abstractmethod
    def sample(self, shape: Tuple[int, ...] = ()) -> Array:
        """Draw random samples."""
        pass

    @abstractmethod
    def log_prob(self, x: Array) -> Array:
        """Compute log probability."""
        pass

    @abstractmethod
    def entropy(self) -> float:
        """Compute entropy."""
        pass

    @abstractmethod
    def mean(self) -> Array:
        """Compute mean."""
        pass

    @abstractmethod
    def var(self) -> Array:
        """Compute variance."""
        pass


# ================================================================
# CONTINUOUS DISTRIBUTIONS
# ================================================================

class Gaussian(Distribution):
    """
    Gaussian (Normal) distribution: N(mu, sigma^2).
    
    PDF:
        p(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2 * sigma^2))
    
    Log PDF:
        log p(x) = -0.5 * log(2*pi) - log(sigma) - (x - mu)^2 / (2 * sigma^2)
    
    Multivariate:
        p(x) = |2*pi*Sigma|^(-1/2) * exp(-0.5 * (x-mu)^T Sigma^-1 (x-mu))
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma
        self.sigma_sq = sigma ** 2

    def pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mu) / self.sigma
        return np.exp(-0.5 * z ** 2) / (self.sigma * np.sqrt(2 * np.pi))

    def log_pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        return -0.5 * np.log(2 * np.pi) - np.log(self.sigma) - 0.5 * ((x - self.mu) / self.sigma) ** 2

    def sample(self, shape: Tuple[int, ...] = ()) -> Array:
        return np.random.normal(self.mu, self.sigma, size=shape)

    def entropy(self) -> float:
        return 0.5 * (1 + np.log(2 * np.pi)) + np.log(self.sigma)

    def mean(self) -> Array:
        return np.array(self.mu)

    def var(self) -> Array:
        return np.array(self.sigma_sq)

    def kl_divergence(self, other: "Gaussian") -> float:
        """KL(N(mu1,sigma1) || N(mu2,sigma2))."""
        return (
            np.log(other.sigma / self.sigma)
            + (self.sigma_sq + (self.mu - other.mu) ** 2) / (2 * other.sigma_sq)
            - 0.5
        )


class MultivariateGaussian(Distribution):
    """
    Multivariate Gaussian: N(mu, Sigma).
    
    Log PDF:
        log p(x) = -d/2 * log(2*pi) - 0.5 * log|Sigma| - 0.5 * (x-mu)^T Sigma^-1 (x-mu)
    """

    def __init__(self, mu: Array, cov: Array):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.d = len(mu)
        self.cov_inv = np.linalg.inv(cov)
        self._log_det = np.log(max(np.linalg.det(cov), 1e-300))

    def log_pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        diff = x - self.mu
        mahal = np.sum(diff @ self.cov_inv * diff, axis=-1)
        return -0.5 * (self.d * np.log(2 * np.pi) + self._log_det + mahal)

    def pdf(self, x: Array) -> Array:
        return np.exp(self.log_pdf(x))

    def sample(self, shape: Tuple[int, ...] = ()) -> Array:
        return np.random.multivariate_normal(self.mu, self.cov, size=shape)

    def entropy(self) -> float:
        return 0.5 * (self.d * (1 + np.log(2 * np.pi)) + self._log_det)

    def mean(self) -> Array:
        return self.mu.copy()

    def var(self) -> Array:
        return np.diag(self.cov)

    def kl_divergence(self, other: "MultivariateGaussian") -> float:
        """KL(p || q) for two multivariate Gaussians."""
        cov_q_inv = np.linalg.inv(other.cov)
        tr = np.trace(cov_q_inv @ self.cov)
        diff = other.mu - self.mu
        quad = diff @ cov_q_inv @ diff
        log_det = np.log(np.linalg.det(other.cov)) - np.log(np.linalg.det(self.cov))
        return 0.5 * (tr + quad - self.d + log_det)


class Beta(Distribution):
    """
    Beta distribution: Beta(alpha, beta).
    
    PDF:
        p(x) = x^(a-1) * (1-x)^(b-1) / B(a,b)
    
    where B(a,b) = Gamma(a)*Gamma(b) / Gamma(a+b) is the Beta function.
    
    Support: x in [0, 1]
    Used in: Bayesian inference (conjugate prior for Bernoulli/Binomial),
             Thompson sampling, Dirichlet components.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self._log_beta_fn = (
            np.math.lgamma(alpha) + np.math.lgamma(beta) - np.math.lgamma(alpha + beta)
        )

    def pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 1e-10, 1 - 1e-10)
        log_p = (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log(1 - x) - self._log_beta_fn
        return np.exp(log_p)

    def log_pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 1e-10, 1 - 1e-10)
        return (self.alpha - 1) * np.log(x) + (self.beta - 1) * np.log(1 - x) - self._log_beta_fn

    def sample(self, shape: Tuple[int, ...] = ()) -> Array:
        return np.random.beta(self.alpha, self.beta, size=shape)

    def entropy(self) -> float:
        a, b = self.alpha, self.beta
        return (
            self._log_beta_fn
            - (a - 1) * np.math.digamma(a)
            - (b - 1) * np.math.digamma(b)
            + (a + b - 2) * np.math.digamma(a + b)
        )

    def mean(self) -> Array:
        return np.array(self.alpha / (self.alpha + self.beta))

    def var(self) -> Array:
        a, b = self.alpha, self.beta
        return np.array(a * b / ((a + b) ** 2 * (a + b + 1)))


class Uniform(Distribution):
    """Continuous Uniform distribution: U(a, b)."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low = low
        self.high = high

    def pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        p = np.zeros_like(x)
        mask = (x >= self.low) & (x <= self.high)
        p[mask] = 1.0 / (self.high - self.low)
        return p

    def log_pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        log_p = np.full_like(x, -np.inf)
        mask = (x >= self.low) & (x <= self.high)
        log_p[mask] = -np.log(self.high - self.low)
        return log_p

    def sample(self, shape: Tuple[int, ...] = ()) -> Array:
        return np.random.uniform(self.low, self.high, size=shape)

    def entropy(self) -> float:
        return np.log(self.high - self.low)

    def mean(self) -> Array:
        return np.array((self.low + self.high) / 2.0)

    def var(self) -> Array:
        return np.array((self.high - self.low) ** 2 / 12.0)


# ================================================================
# DISCRETE DISTRIBUTIONS
# ================================================================

class Categorical(Distribution):
    """
    Categorical distribution: Cat(p1, ..., pK).
    
    PMF:
        P(x = k) = p_k
    
    Support: x in {0, 1, ..., K-1}
    Parameters: p = (p1, ..., pK) with sum(p) = 1
    
    Used in: Classification, language modeling (next-token prediction).
    """

    def __init__(self, probs: Optional[Array] = None, logits: Optional[Array] = None):
        if logits is not None:
            logits = np.asarray(logits, dtype=np.float64)
            self.logits = logits
            # Numerically stable softmax
            x = logits - np.max(logits)
            e = np.exp(x)
            self.probs = e / np.sum(e)
        elif probs is not None:
            self.probs = np.asarray(probs, dtype=np.float64)
            self.probs = self.probs / np.sum(self.probs)
            self.logits = np.log(np.maximum(self.probs, 1e-12))
        else:
            raise ValueError("Must provide either probs or logits")
        
        self.K = len(self.probs)
        self.log_probs = np.log(np.maximum(self.probs, 1e-12))

    def pmf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=int)
        return self.probs[x]

    def log_pmf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=int)
        return self.log_probs[x]

    def log_prob(self, x: Array) -> Array:
        return self.log_pmf(x)

    def sample(self, shape: Tuple[int, ...] = ()) -> Array:
        return np.random.choice(self.K, size=shape, p=self.probs)

    def entropy(self) -> float:
        return float(-np.sum(self.probs * self.log_probs))

    def mean(self) -> Array:
        return np.array(np.sum(np.arange(self.K) * self.probs))

    def var(self) -> Array:
        m = self.mean()
        return np.array(np.sum(self.probs * (np.arange(self.K) - m) ** 2))

    def kl_divergence(self, other: "Categorical") -> float:
        """KL(p || q) = sum p * log(p/q)."""
        return float(np.sum(self.probs * (self.log_probs - other.log_probs)))


class Multinomial(Distribution):
    """
    Multinomial distribution: Multi(n, p1, ..., pK).
    
    PMF:
        P(x1=k1, ..., xK=kK) = n! / (k1! * ... * kK!) * p1^k1 * ... * pK^kK
        where sum(ki) = n
    
    Generalizes Bernoulli (K=2) and Categorical (n=1).
    """

    def __init__(self, n: int, probs: Array):
        self.n = n
        self.probs = np.asarray(probs, dtype=np.float64)
        self.probs = self.probs / np.sum(self.probs)
        self.K = len(self.probs)
        self.log_probs = np.log(np.maximum(self.probs, 1e-12))

    def log_pmf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        from math import lgamma
        log_coeff = lgamma(self.n + 1) - np.sum([lgamma(int(xi) + 1) for xi in x])
        log_probs = np.sum(x * self.log_probs)
        return log_coeff + log_probs

    def log_prob(self, x: Array) -> Array:
        return self.log_pmf(x)

    def sample(self, shape: Tuple[int, ...] = ()) -> Array:
        return np.random.multinomial(self.n, self.probs, size=shape)

    def entropy(self) -> float:
        """Entropy of multinomial (approximate for large n)."""
        return -self.n * np.sum(self.probs * self.log_probs)

    def mean(self) -> Array:
        return self.n * self.probs

    def var(self) -> Array:
        return self.n * self.probs * (1 - self.probs)


class Bernoulli(Distribution):
    """
    Bernoulli distribution: Ber(p).
    
    PMF: P(x) = p^x * (1-p)^(1-x)
    Support: x in {0, 1}
    
    Special case of Categorical with K=2.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def pmf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        return self.p ** x * (1 - self.p) ** (1 - x)

    def log_pmf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        p_safe = np.clip(self.p, 1e-12, 1 - 1e-12)
        return x * np.log(p_safe) + (1 - x) * np.log(1 - p_safe)

    def log_prob(self, x: Array) -> Array:
        return self.log_pmf(x)

    def sample(self, shape: Tuple[int, ...] = ()) -> Array:
        return (np.random.random(shape) < self.p).astype(np.float64)

    def entropy(self) -> float:
        p_safe = np.clip(self.p, 1e-12, 1 - 1e-12)
        return -p_safe * np.log(p_safe) - (1 - p_safe) * np.log(1 - p_safe)

    def mean(self) -> Array:
        return np.array(self.p)

    def var(self) -> Array:
        return np.array(self.p * (1 - self.p))


class Dirichlet(Distribution):
    """
    Dirichlet distribution: Dir(alpha_1, ..., alpha_K).
    
    PDF:
        p(x) = (1/B(alpha)) * prod_i x_i^(alpha_i - 1)
    
    Support: x in simplex {x: x_i >= 0, sum(x_i) = 1}
    
    B(alpha) = prod Gamma(alpha_i) / Gamma(sum alpha_i)
    
    Properties:
        - Mean: E[x_i] = alpha_i / sum(alpha_j)
        - Variance: Var[x_i] = alpha_i*(alpha_0 - alpha_i) / (alpha_0^2 * (alpha_0 + 1))
          where alpha_0 = sum(alpha_j)
        - Conjugate prior for Categorical/Multinomial
        - Entropy has a closed form via digamma function
    
    Used in: LDA (Latent Dirichlet Allocation), Bayesian inference,
             topic modeling, attention distributions.
    """

    def __init__(self, alpha: Array):
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.K = len(self.alpha)
        self.alpha_0 = np.sum(self.alpha)
        self._log_B = (
            np.sum(np.array([np.math.lgamma(a) for a in self.alpha]))
            - np.math.lgamma(self.alpha_0)
        )

    def pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 1e-10, 1.0)
        log_p = -self._log_B + np.sum((self.alpha - 1) * np.log(x), axis=-1)
        return np.exp(log_p)

    def log_pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 1e-10, 1.0)
        return -self._log_B + np.sum((self.alpha - 1) * np.log(x), axis=-1)

    def log_prob(self, x: Array) -> Array:
        return self.log_pdf(x)

    def sample(self, shape: Tuple[int, ...] = ()) -> Array:
        return np.random.dirichlet(self.alpha, size=shape)

    def entropy(self) -> float:
        a0 = self.alpha_0
        K = self.K
        digamma_a0 = np.math.digamma(a0)
        return (
            self._log_B
            + (a0 - K) * digamma_a0
            - np.sum([(a - 1) * np.math.digamma(a) for a in self.alpha])
        )

    def mean(self) -> Array:
        return self.alpha / self.alpha_0

    def var(self) -> Array:
        a0 = self.alpha_0
        return self.alpha * (a0 - self.alpha) / (a0 ** 2 * (a0 + 1))

    def kl_divergence(self, other: "Dirichlet") -> float:
        """
        KL(Dir(alpha) || Dir(beta)):
            = log B(beta) - log B(alpha)
              + sum (alpha_i - beta_i) * (psi(alpha_i) - psi(alpha_0))
        """
        psi_a0 = np.math.digamma(self.alpha_0)
        psi_b0 = np.math.digamma(other.alpha_0)
        psi_diff = np.array([np.math.digamma(a) - psi_a0 for a in self.alpha])
        return (
            other._log_B - self._log_B
            + np.sum((self.alpha - other.alpha) * psi_diff)
            + np.sum((self.alpha_0 - other.alpha_0) * psi_b0)
        )


class MixtureOfGaussians(Distribution):
    """
    Gaussian Mixture Model: p(x) = sum_k pi_k * N(x | mu_k, sigma_k^2).
    
    Parameters:
        - weights (pi): mixing coefficients, sum(pi) = 1
        - means (mu): component means
        - stds (sigma): component standard deviations
    
    PMF: p(x) = sum_k pi_k * N(x | mu_k, sigma_k^2)
    Log-PMF (log-sum-exp for stability): log p(x) = logsumexp_k(log(pi_k) + log N(x|mu_k,sigma_k^2))
    
    Used in: Density estimation, clustering, EM algorithm, VAE latent space.
    """

    def __init__(self, weights: Array, means: Array, stds: Array):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.stds = np.asarray(stds, dtype=np.float64)
        self.K = len(weights)
        self.components = [Gaussian(m, s) for m, s in zip(means, stds)]
        self.log_weights = np.log(np.maximum(self.weights, 1e-12))

    def pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        # Log-sum-exp trick for numerical stability
        log_probs = np.array([
            self.log_weights[k] + self.components[k].log_pdf(x)
            for k in range(self.K)
        ])
        max_log = np.max(log_probs, axis=0)
        return np.exp(max_log) * np.sum(np.exp(log_probs - max_log), axis=0)

    def log_pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        log_probs = np.array([
            self.log_weights[k] + self.components[k].log_pdf(x)
            for k in range(self.K)
        ])
        max_log = np.max(log_probs, axis=0)
        return max_log + np.log(np.sum(np.exp(log_probs - max_log), axis=0))

    def log_prob(self, x: Array) -> Array:
        return self.log_pdf(x)

    def sample(self, shape: Tuple[int, ...] = ()) -> Array:
        # First sample component, then sample from that component
        components = np.random.choice(self.K, size=shape, p=self.weights)
        samples = np.zeros(shape, dtype=np.float64)
        for k in range(self.K):
            mask = components == k
            count = np.sum(mask)
            if count > 0:
                samples[mask] = self.components[k].sample(shape=(count,))
        return samples

    def entropy(self) -> float:
        """Approximate entropy via Monte Carlo."""
        samples = self.sample(shape=(10000,))
        return float(-np.mean(self.log_pdf(samples)))

    def mean(self) -> Array:
        return np.sum(self.weights * self.means)

    def var(self) -> Array:
        m = self.mean()
        return np.sum(self.weights * (self.stds ** 2 + self.means ** 2)) - m ** 2


class Empirical(Distribution):
    """
    Empirical distribution from samples.
    
    PDF is a sum of Dirac deltas:
        p(x) = (1/N) * sum_i delta(x - x_i)
    
    Mean: sample mean = (1/N) * sum x_i
    Variance: sample variance = (1/N) * sum (x_i - mean)^2
    """

    def __init__(self, samples: Array):
        self.samples = np.asarray(samples, dtype=np.float64)
        self.n = len(self.samples)

    def pdf(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        # Use kernel density estimation for continuous output
        bandwidth = np.std(self.samples) * (self.n ** (-1/5))  # Silverman's rule
        if bandwidth < 1e-10:
            bandwidth = 1.0
        kde = np.sum(np.exp(-0.5 * ((x[:, None] - self.samples[None, :]) / bandwidth) ** 2), axis=1)
        return kde / (self.n * bandwidth * np.sqrt(2 * np.pi))

    def log_pdf(self, x: Array) -> Array:
        return np.log(np.maximum(self.pdf(x), 1e-300))

    def log_prob(self, x: Array) -> Array:
        return self.log_pdf(x)

    def sample(self, shape: Tuple[int, ...] = ()) -> Array:
        indices = np.random.randint(0, self.n, size=shape)
        return self.samples[indices]

    def entropy(self) -> float:
        """Plug-in entropy estimator with bias correction."""
        hist, _ = np.histogram(self.samples, bins=50, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist)) * (self.samples.max() - self.samples.min()) / 50

    def mean(self) -> Array:
        return np.mean(self.samples)

    def var(self) -> Array:
        return np.var(self.samples)
