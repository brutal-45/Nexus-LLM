"""
Nexus Math Library
=====================
Complete mathematical foundations built from scratch.

Submodules:
    - tensor: Custom Tensor class with shape manipulation and broadcasting
    - linalg: Linear algebra (matmul, Strassen, eigen, SVD, QR, Cholesky, ...)
    - autodiff: Automatic differentiation (forward/reverse mode, computational graph)
    - distributions: Probability distributions (Gaussian, Categorical, Dirichlet, ...)
    - sampling: Sampling methods (ancestral, top-k, top-p, Gumbel-Softmax, ...)
    - information: Information theory (entropy, KL, cross-entropy, mutual info, ...)
    - bayesian: Bayesian inference fundamentals
    - numerical: Numerical stability (gradient clipping, loss scaling, ...)
"""

from .tensor import Tensor
from .linalg import (
    matmul, matmul_strassen, matmul_tiled, batch_matmul,
    eigen_decomposition, svd, qr_factorization, cholesky,
    hadamard_product, kronecker_product,
    trace, determinant, inverse, pinv,
    lu_decomposition, solve_triangular, solve,
    norm, spectral_norm, condition_number,
    transpose, permute, reshape, einsum,
)
from .autodiff import (
    Variable, ComputationalGraph,
    forward_mode_ad, reverse_mode_ad, backprop,
    jacobian, hessian, grad, value_and_grad,
)
from .distributions import (
    Distribution, Gaussian, Categorical, Multinomial,
    Dirichlet, Bernoulli, Beta, Uniform,
    MixtureOfGaussians, Empirical,
)
from .sampling import (
    ancestral_sample, top_k_sample, top_p_sample,
    temperature_sample, gumbel_softmax_sample,
    beam_search_sample, repetition_penalty_sample,
    Sampler, Random,
)
from .information import (
    entropy, cross_entropy, kl_divergence,
    mutual_information, perplexity, js_divergence,
    conditional_entropy, binary_cross_entropy,
)
from .bayesian import (
    BayesianPosterior, GaussianPosterior, MCMCSampler,
    MetropolisHastings, GibbsSampler, VariationalInference,
)
from .numerical import (
    gradient_clip_norm, gradient_clip_value,
    dynamic_loss_scaling, static_loss_scaling,
    check_numerical_stability, safe_softmax, safe_log,
)

__all__ = [
    # Tensor
    "Tensor",
    # Linalg
    "matmul", "matmul_strassen", "matmul_tiled", "batch_matmul",
    "eigen_decomposition", "svd", "qr_factorization", "cholesky",
    "hadamard_product", "kronecker_product",
    "trace", "determinant", "inverse", "pinv",
    "lu_decomposition", "solve_triangular", "solve",
    "norm", "spectral_norm", "condition_number",
    "transpose", "permute", "reshape", "einsum",
    # Autodiff
    "Variable", "ComputationalGraph",
    "forward_mode_ad", "reverse_mode_ad", "backprop",
    "jacobian", "hessian", "grad", "value_and_grad",
    # Distributions
    "Distribution", "Gaussian", "Categorical", "Multinomial",
    "Dirichlet", "Bernoulli", "Beta", "Uniform",
    "MixtureOfGaussians", "Empirical",
    # Sampling
    "ancestral_sample", "top_k_sample", "top_p_sample",
    "temperature_sample", "gumbel_softmax_sample",
    "beam_search_sample", "repetition_penalty_sample",
    "Sampler", "Random",
    # Information theory
    "entropy", "cross_entropy", "kl_divergence",
    "mutual_information", "perplexity", "js_divergence",
    "conditional_entropy", "binary_cross_entropy",
    # Bayesian
    "BayesianPosterior", "GaussianPosterior", "MCMCSampler",
    "MetropolisHastings", "GibbsSampler", "VariationalInference",
    # Numerical
    "gradient_clip_norm", "gradient_clip_value",
    "dynamic_loss_scaling", "static_loss_scaling",
    "check_numerical_stability", "safe_softmax", "safe_log",
]
