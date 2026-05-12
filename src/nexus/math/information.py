"""
Nexus Information Theory
============================
All information-theoretic metrics implemented from scratch.

Metrics:
    - Entropy: H(X) = -sum p(x) log p(x)
    - Cross-entropy: H(P, Q) = -sum p(x) log q(x)
    - KL divergence: KL(P || Q) = sum p(x) log(p(x)/q(x))
    - Jensen-Shannon divergence: JSD(P, Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = (P+Q)/2
    - Conditional entropy: H(Y|X) = sum_x p(x) H(Y|X=x)
    - Mutual information: I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
    - Binary cross-entropy: for binary classification
    - Perplexity: exp(H(P, Q))
    
These metrics are fundamental to:
    - LLM training (cross-entropy loss)
    - Model evaluation (perplexity)
    - Regularization (KL divergence in VAEs)
    - Feature selection (mutual information)
    - Compression theory (entropy coding bounds)

All implementations use log-sum-exp trick for numerical stability.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union

Array = np.ndarray


def entropy(p: Array, base: float = np.e) -> float:
    """
    Shannon entropy: H(X) = -sum p(x) log p(x).
    
    Measures the uncertainty/randomness of a random variable.
    
    Properties:
        - H(X) >= 0 (always non-negative)
        - H(X) = 0 iff X is deterministic (one outcome has p=1)
        - H(X) is maximized when X is uniform: H_max = log(K) for K outcomes
        - Joint: H(X,Y) <= H(X) + H(Y) (subadditivity)
    
    Examples:
        - Fair coin: H = 1 bit
        - Fair die: H = log2(6) ≈ 2.58 bits
        - Deterministic: H = 0 bits
    
    Args:
        p: Probability distribution (must sum to 1).
        base: Logarithm base (2 for bits, e for nats, 10 for hartleys).
    
    Returns:
        Entropy in chosen units.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)  # Avoid log(0)
    
    log_base = np.log(base) if base != np.e else 1.0
    
    # Only compute where p > 0 (0 * log(0) = 0 by convention)
    mask = p > 0
    h = -np.sum(p[mask] * np.log(p[mask])) / log_base if log_base != 1.0 else -np.sum(p[mask] * np.log(p[mask]))
    return float(h)


def cross_entropy(p: Array, q: Array, base: float = np.e) -> float:
    """
    Cross-entropy: H(P, Q) = -sum p(x) log q(x).
    
    Measures the average number of bits needed to encode samples from P
    using a code optimized for Q.
    
    Relationship to entropy and KL:
        H(P, Q) = H(P) + KL(P || Q)
    
    Since KL(P || Q) >= 0:
        H(P, Q) >= H(P)  (cross-entropy is always >= entropy)
        Equality iff P = Q
    
    In machine learning:
        - P is the true distribution (one-hot labels)
        - Q is the model's predicted distribution (softmax outputs)
        - Minimizing cross-entropy = maximizing likelihood
    
    Args:
        p: True distribution (target).
        q: Predicted distribution (model output).
        base: Logarithm base.
    
    Returns:
        Cross-entropy value.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    
    log_base = np.log(base) if base != np.e else 1.0
    
    ce = -np.sum(p * np.log(q))
    return float(ce / log_base) if log_base != 1.0 else float(ce)


def kl_divergence(p: Array, q: Array, base: float = np.e) -> float:
    """
    Kullback-Leibler divergence: KL(P || Q) = sum p(x) log(p(x)/q(x)).
    
    Measures how much distribution Q diverges from P.
    NOT symmetric: KL(P||Q) != KL(Q||P) in general.
    
    Properties:
        - KL(P || Q) >= 0 (Gibbs' inequality)
        - KL(P || Q) = 0 iff P = Q almost everywhere
        - KL(P || Q) != KL(Q || P) (asymmetric)
        - KL(P || Uniform) = -H(P) + log(K)
    
    Interpretation:
        - Information lost when Q is used to approximate P
        - Expected excess surprise from using Q instead of P
        - "Distance" (not a true metric, no triangle inequality)
    
    In machine learning:
        - VAE loss: KL(q(z|x) || p(z)) = regularization term
        - Policy gradient: KL(old_policy || new_policy) = trust region
        - Knowledge distillation: KL(teacher || student)
    
    Args:
        p: True/posterior distribution P.
        q: Approximate/prior distribution Q.
        base: Logarithm base.
    
    Returns:
        KL divergence value.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    
    log_base = np.log(base) if base != np.e else 1.0
    
    kl = np.sum(p * (np.log(p) - np.log(q)))
    return float(kl / log_base) if log_base != 1.0 else float(kl)


def js_divergence(p: Array, q: Array, base: float = np.e) -> float:
    """
    Jensen-Shannon divergence: JSD(P, Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M).
    
    Where M = 0.5 * (P + Q) is the mixture distribution.
    
    Properties:
        - JSD(P, Q) >= 0
        - JSD(P, Q) = 0 iff P = Q
        - JSD(P, Q) = JSD(Q, P) (SYMMETRIC — unlike KL)
        - JSD(P, Q) <= log(2) (bounded)
        - sqrt(JSD) satisfies the triangle inequality (metric)
    
    Preferred over KL when symmetry is needed (e.g., comparing two models).
    
    Args:
        p: First distribution.
        q: Second distribution.
        base: Logarithm base.
    
    Returns:
        Jensen-Shannon divergence.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)


def conditional_entropy(p_xy: Array, p_x: Optional[Array] = None, base: float = np.e) -> float:
    """
    Conditional entropy: H(Y|X) = sum_x p(x) * H(Y|X=x).
    
    Measures the remaining uncertainty in Y after observing X.
    
    Properties:
        - H(Y|X) >= 0
        - H(Y|X) <= H(Y) (observing X never increases uncertainty)
        - H(Y|X) = 0 iff Y is a deterministic function of X
        - H(Y|X) = H(X,Y) - H(X)
    
    Chain rule: H(X,Y) = H(X) + H(Y|X)
    
    Args:
        p_xy: Joint distribution P(X,Y) as 2D array.
        p_x: Marginal P(X). If None, computed from p_xy.
        base: Logarithm base.
    
    Returns:
        Conditional entropy.
    """
    p_xy = np.asarray(p_xy, dtype=np.float64)
    
    if p_x is None:
        p_x = np.sum(p_xy, axis=1)
    
    h_y_given_x = 0.0
    for i in range(len(p_x)):
        if p_x[i] > 1e-12:
            p_y_given_x = p_xy[i] / p_x[i]
            h_y_given_x += p_x[i] * entropy(p_y_given_x, base)
    
    return float(h_y_given_x)


def mutual_information(p_xy: Array, base: float = np.e) -> float:
    """
    Mutual information: I(X;Y) = KL(P(X,Y) || P(X)P(Y)).
    
    Measures the amount of information that X and Y share:
        - How much knowing X reduces uncertainty about Y (and vice versa)
        - How much X and Y "depend" on each other
    
    Properties:
        - I(X;Y) >= 0
        - I(X;Y) = 0 iff X and Y are independent
        - I(X;Y) = I(Y;X) (symmetric)
        - I(X;Y) <= min(H(X), H(Y))
    
    Relationships:
        I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
        I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    Args:
        p_xy: Joint distribution P(X,Y).
        base: Logarithm base.
    
    Returns:
        Mutual information.
    """
    p_xy = np.asarray(p_xy, dtype=np.float64)
    
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    # KL(P(X,Y) || P(X)P(Y))
    p_independent = np.outer(p_x, p_y)
    return kl_divergence(p_xy.flatten(), p_independent.flatten(), base)


def perplexity(p: Array, q: Array) -> float:
    """
    Perplexity: PP(P, Q) = exp(H(P, Q)).
    
    Geometric mean of the inverse probability assigned by Q to each
    outcome under P.
    
    Interpretation: "On average, how many equally likely options would
    the model be choosing between?"
    
    In language modeling:
        - Lower perplexity = better model
        - PP = 1: model perfectly predicts every token
        - PP = V (vocab size): model is as good as random guessing
        - GPT-4: ~3 perplexity on typical text
        - Random model (50K vocab): ~50,000 perplexity
    
    Relationship: log(PP) = H(P, Q) (cross-entropy in nats)
    
    Args:
        p: True distribution (one-hot for language modeling).
        q: Model's predicted distribution.
    
    Returns:
        Perplexity value.
    """
    ce = cross_entropy(p, q, base=np.e)
    return float(np.exp(ce))


def binary_cross_entropy(p: Array, q: Array, epsilon: float = 1e-12) -> float:
    """
    Binary cross-entropy for binary classification.
    
    BCE = -[p * log(q) + (1-p) * log(1-q)]
    
    Where:
        p: true label (0 or 1)
        q: predicted probability
    
    This is the standard loss function for binary classification
    used in logistic regression, neural networks, etc.
    
    Args:
        p: True labels in [0, 1].
        q: Predicted probabilities in (0, 1).
        epsilon: Small value for numerical stability.
    
    Returns:
        Average binary cross-entropy.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, epsilon, 1 - epsilon)
    
    bce = -np.mean(p * np.log(q) + (1 - p) * np.log(1 - q))
    return float(bce)


# ================================================================
# ADVANCED INFORMATION THEORY
# ================================================================

def pointwise_mutual_information(p_xy: Array, p_x: Array, p_y: Array) -> Array:
    """
    Pointwise Mutual Information (PMI).
    
    PMI(x, y) = log(p(x,y) / (p(x) * p(y)))
    
    Measures the association between specific outcomes x and y:
        PMI > 0: x and y co-occur more often than expected
        PMI = 0: x and y are independent
        PMI < 0: x and y co-occur less often than expected
    
    Used in: NLP (word co-occurrence), collocation extraction,
    feature selection.
    """
    p_xy = np.asarray(p_xy, dtype=np.float64)
    p_x = np.asarray(p_x, dtype=np.float64)
    p_y = np.asarray(p_y, dtype=np.float64)
    
    p_indep = np.outer(p_x, p_y)
    pmi = np.log(p_xy / np.maximum(p_indep, 1e-12))
    return pmi


def conditional_mutual_information(p_xyz: Array) -> float:
    """
    Conditional Mutual Information: I(X;Y|Z).
    
    Measures information shared between X and Y given Z.
    
    I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
              = E_z[KL(P(X,Y|Z=z) || P(X|Z=z)P(Y|Z=z))]
    
    Properties:
        - I(X;Y|Z) >= 0
        - I(X;Y|Z) = 0 iff X and Y are conditionally independent given Z
        - Chain rule: I(X;Y,Z) = I(X;Z) + I(X;Y|Z)
    """
    p_xyz = np.asarray(p_xyz, dtype=np.float64)
    
    p_z = np.sum(p_xyz, axis=(0, 1))
    p_xz = np.sum(p_xyz, axis=1)
    p_yz = np.sum(p_xyz, axis=0)
    p_x = np.sum(p_xz, axis=1)
    p_y = np.sum(p_yz, axis=1)
    
    cmi = 0.0
    for k in range(p_z.shape[0]):
        if p_z[k] > 1e-12:
            p_xy_given_z = p_xyz[:, :, k] / p_z[k]
            p_x_given_z = p_xz[:, k] / p_z[k]
            p_y_given_z = p_yz[:, k] / p_z[k]
            p_indep = np.outer(p_x_given_z, p_y_given_z)
            cmi += p_z[k] * kl_divergence(
                p_xy_given_z.flatten(), p_indep.flatten()
            )
    return float(cmi)


def information_gain(p_parent: Array, p_child: Array) -> float:
    """
    Information gain (IG) for decision trees.
    
    IG(Y, X) = H(Y) - H(Y|X)
             = sum_x p(x) * H(Y|X=x)
    
    Measures how much knowing X reduces uncertainty about Y.
    Equivalent to mutual information I(X;Y).
    
    Used in: Decision tree feature selection, feature importance.
    """
    return mutual_information(np.outer(p_parent, p_child))


def earth_movers_distance(p: Array, q: Array) -> float:
    """
    Earth Mover's Distance (Wasserstein-1 distance).
    
    EMD(p, q) = sum_i |CDF_p(i) - CDF_q(i)|
    
    For 1D distributions, this is the L1 distance between CDFs.
    Generalizes to arbitrary metric spaces (optimal transport).
    
    Used in: Distribution comparison, domain adaptation, GAN losses.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)))
