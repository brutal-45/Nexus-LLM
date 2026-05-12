"""
Nexus Bayesian Inference
=============================
Fundamentals of Bayesian inference implemented from scratch.

Components:
    - BayesianPosterior: Generic posterior inference framework
    - GaussianPosterior: Closed-form Bayesian inference for Gaussian models
    - MCMCSampler: Markov Chain Monte Carlo sampling
    - MetropolisHastings: Metropolis-Hastings MCMC algorithm
    - GibbsSampler: Gibbs sampling for conjugate models
    - VariationalInference: Mean-field variational inference

Bayesian Inference Fundamentals:
    Prior:          p(θ)           - Beliefs before seeing data
    Likelihood:     p(D|θ)        - Probability of data given parameters
    Posterior:      p(θ|D) ∝ p(D|θ) * p(θ)  - Updated beliefs (Bayes' theorem)
    Evidence:       p(D) = ∫ p(D|θ) p(θ) dθ   - Normalizing constant
    Predictive:     p(D_new|D) = ∫ p(D_new|θ) p(θ|D) dθ

MCMC Sampling:
    When the posterior is intractable, MCMC generates samples from p(θ|D)
    without computing the normalizing constant. The samples approximate
    the posterior distribution.

Variational Inference:
    Approximates p(θ|D) with a simpler distribution q(θ) by minimizing
    KL(q(θ) || p(θ|D)). Faster than MCMC but may underfit.

Reference:
    - Gelman et al., "Bayesian Data Analysis" (3rd ed.)
    - Murphy, "Machine Learning: A Probabilistic Perspective"
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

Array = np.ndarray


# ================================================================
# GENERIC BAYESIAN POSTERIOR
# ================================================================

class BayesianPosterior:
    """
    Generic Bayesian posterior computation.
    
    Bayes' theorem: p(θ|D) = p(D|θ) * p(θ) / p(D)
    
    In practice, p(D) is often intractable (the "evidence" or
    "marginal likelihood"). We work with the unnormalized posterior:
        p(θ|D) ∝ p(D|θ) * p(θ)
    """

    def __init__(
        self,
        log_likelihood: Callable[[Array], float],
        log_prior: Callable[[Array], float],
        log_posterior_unnorm: Optional[Callable[[Array], float]] = None,
    ):
        """
        Args:
            log_likelihood: log p(D | θ).
            log_prior: log p(θ).
            log_posterior_unnorm: Optional pre-computed log posterior ∝ log p(θ|D).
        """
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        
        if log_posterior_unnorm is not None:
            self.log_posterior_unnorm = log_posterior_unnorm
        else:
            self.log_posterior_unnorm = lambda theta: (
                log_likelihood(theta) + log_prior(theta)
            )

    def posterior(self, theta: Array) -> float:
        """Unnormalized posterior density."""
        return self.log_posterior_unnorm(theta)

    def posterior_normalized(
        self,
        theta_grid: Array,
    ) -> Tuple[Array, Array]:
        """
        Compute normalized posterior on a grid (for 1D/2D).
        
        Uses log-sum-exp trick for numerical stability.
        """
        log_posteriors = np.array([self.log_posterior_unnorm(t) for t in theta_grid])
        
        # Log-sum-exp trick
        max_lp = np.max(log_posteriors)
        log_evidence = max_lp + np.log(np.sum(np.exp(log_posteriors - max_lp)))
        
        # Normalize
        log_posteriors_norm = log_posteriors - log_evidence
        posteriors = np.exp(log_posteriors_norm)
        
        return theta_grid, posteriors

    def predict(
        self,
        likelihood_fn: Callable[[Array, Array], float],
        thetas: Array,
        x_new: Optional[Array] = None,
    ) -> float:
        """
        Bayesian posterior predictive: p(D_new | D) = ∫ p(D_new|θ) p(θ|D) dθ.
        
        Monte Carlo approximation:
            p(D_new|D) ≈ (1/N) * sum_i p(D_new | θ_i)
        where θ_i ~ p(θ|D).
        """
        if x_new is not None:
            log_preds = [likelihood_fn(theta, x_new) for theta in thetas]
        else:
            log_preds = [self.log_likelihood(theta) for theta in thetas]
        
        # Log-sum-exp
        max_lp = np.max(log_preds)
        return float(np.exp(max_lp) * np.mean(np.exp(np.array(log_preds) - max_lp)))


# ================================================================
# GAUSSIAN POSTERIOR (CLOSED-FORM)
# ================================================================

class GaussianPosterior:
    """
    Closed-form Bayesian inference for Gaussian models.
    
    Model:
        Prior:      μ ~ N(μ_0, σ_0^2)
        Likelihood: x | μ ~ N(μ, σ^2)
    
    Posterior (conjugate):
        μ | x ~ N(μ_n, σ_n^2)
    
    where:
        1/σ_n^2 = 1/σ_0^2 + n/σ^2        (precision combines)
        μ_n = σ_n^2 * (μ_0/σ_0^2 + sum(x_i)/σ^2)  (precision-weighted mean)
    
    For multivariate:
        Σ_n^(-1) = Σ_0^(-1) + n * Σ^(-1)
        μ_n = Σ_n * (Σ_0^(-1) @ μ_0 + n * Σ^(-1) @ x̄)
    
    The Gaussian is its own conjugate prior — one of the most important
    results in Bayesian statistics.
    """

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        likelihood_var: float = 1.0,
    ):
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.likelihood_var = likelihood_var

    def update(self, data: Array) -> Dict[str, float]:
        """
        Compute posterior parameters after observing data.
        
        Args:
            data: Observed data points.
        
        Returns:
            Dictionary with posterior mean, variance, and precision.
        """
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        x_bar = np.mean(data)
        
        prior_prec = 1.0 / self.prior_var
        lik_prec = 1.0 / self.likelihood_var
        
        # Posterior precision
        post_prec = prior_prec + n * lik_prec
        post_var = 1.0 / post_prec
        
        # Posterior mean (precision-weighted average)
        post_mean = post_var * (prior_prec * self.prior_mean + lik_prec * np.sum(data))
        
        return {
            "posterior_mean": float(post_mean),
            "posterior_var": float(post_var),
            "posterior_precision": float(post_prec),
            "prior_precision": float(prior_prec),
            "data_mean": float(x_bar),
            "n_observations": n,
        }

    def posterior_sample(
        self,
        data: Array,
        n_samples: int = 1000,
    ) -> Array:
        """Draw samples from the posterior distribution."""
        post = self.update(data)
        return np.random.normal(
            post["posterior_mean"],
            np.sqrt(post["posterior_var"]),
            size=n_samples,
        )

    def posterior_predictive(
        self,
        data: Array,
        n_samples: int = 1000,
    ) -> Array:
        """
        Draw samples from the posterior predictive distribution.
        
        p(x_new | D) = ∫ N(x_new | μ, σ^2) N(μ | μ_n, σ_n^2) dμ
                     = N(x_new | μ_n, σ^2 + σ_n^2)
        
        The predictive distribution accounts for BOTH observation noise
        AND posterior uncertainty in the parameters.
        """
        post = self.update(data)
        pred_var = self.likelihood_var + post["posterior_var"]
        return np.random.normal(
            post["posterior_mean"],
            np.sqrt(pred_var),
            size=n_samples,
        )


class MultivariateGaussianPosterior:
    """
    Bayesian inference for multivariate Gaussian with known covariance.
    
    Prior:      μ ~ N(μ_0, Σ_0)
    Likelihood: X | μ ~ N(μ, Σ)  (Σ known)
    
    Posterior:
        Σ_n^(-1) = Σ_0^(-1) + n * Σ^(-1)
        μ_n = Σ_n * (Σ_0^(-1) @ μ_0 + n * Σ^(-1) @ X̄)
    """

    def __init__(self, prior_mean: Array, prior_cov: Array, lik_cov: Array):
        self.prior_mean = np.asarray(prior_mean, dtype=np.float64)
        self.prior_cov = np.asarray(prior_cov, dtype=np.float64)
        self.lik_cov = np.asarray(lik_cov, dtype=np.float64)
        self.d = len(prior_mean)
        
        # Precompute precisions
        self.prior_prec = np.linalg.inv(prior_cov)
        self.lik_prec = np.linalg.inv(lik_cov)

    def update(self, data: Array) -> Dict[str, Array]:
        """Compute posterior parameters."""
        data = np.asarray(data, dtype=np.float64)
        n = data.shape[0]
        x_bar = np.mean(data, axis=0)
        
        # Posterior precision
        post_prec = self.prior_prec + n * self.lik_prec
        post_cov = np.linalg.inv(post_prec)
        
        # Posterior mean
        post_mean = post_cov @ (self.prior_prec @ self.prior_mean + n * self.lik_prec @ x_bar)
        
        return {
            "posterior_mean": post_mean,
            "posterior_cov": post_cov,
            "posterior_prec": post_prec,
        }


# ================================================================
# MCMC SAMPLERS
# ================================================================

class MCMCSampler(ABC):
    """Abstract base class for MCMC samplers."""

    @abstractmethod
    def sample(
        self,
        log_posterior: Callable[[Array], float],
        n_samples: int,
        initial: Optional[Array] = None,
    ) -> Tuple[Array, Dict[str, Array]]:
        pass


class MetropolisHastings(MCMCSampler):
    """
    Metropolis-Hastings MCMC algorithm.
    
    The most general MCMC algorithm. Generates samples from any target
    distribution p(θ) that we can evaluate up to a constant.
    
    Algorithm:
        1. Initialize θ_0
        2. For each step t:
           a. Propose θ' ~ q(θ' | θ_t)   (proposal distribution)
           b. Compute acceptance ratio:
              α = min(1, p(θ') / p(θ_t) * q(θ_t | θ') / q(θ' | θ_t))
           c. Accept with probability α: θ_{t+1} = θ' or θ_{t+1} = θ_t
           d. Store θ_{t+1}
    
    For symmetric proposals (q(a|b) = q(b|a)):
        α = min(1, p(θ') / p(θ_t))   (simplified Metropolis)
    
    Properties:
        - Guaranteed to converge to the target distribution (ergodic theorem)
        - Convergence rate depends on proposal distribution
        - Too small proposal step → high acceptance, slow exploration
        - Too large proposal step → low acceptance, many rejections
        - Optimal acceptance rate ≈ 23% for high-dim, 44% for 1D
    
    Args:
        proposal_std: Standard deviation of Gaussian proposal distribution.
        burn_in: Number of initial samples to discard.
        thin: Keep every thin-th sample (to reduce autocorrelation).
    """

    def __init__(
        self,
        proposal_std: float = 0.1,
        burn_in: int = 1000,
        thin: int = 1,
    ):
        self.proposal_std = proposal_std
        self.burn_in = burn_in
        self.thin = thin

    def sample(
        self,
        log_posterior: Callable[[Array], float],
        n_samples: int,
        initial: Optional[Array] = None,
    ) -> Tuple[Array, Dict[str, Array]]:
        """
        Run Metropolis-Hastings MCMC.
        
        Args:
            log_posterior: Unnormalized log posterior function.
            n_samples: Number of samples to return.
            initial: Initial parameter value.
        
        Returns:
            Tuple of (samples, diagnostics) where diagnostics includes
            acceptance rate and log posterior trace.
        """
        d = len(initial) if initial is not None else 1
        if initial is None:
            initial = np.zeros(d)
        
        theta = np.array(initial, dtype=np.float64)
        log_p = log_posterior(theta)
        
        total_steps = self.burn_in + n_samples * self.thin
        samples = []
        log_posts = []
        accepted = 0
        total_proposals = 0
        
        rng = np.random.RandomState(42)
        
        for step in range(total_steps):
            # Propose new state
            theta_proposal = theta + rng.normal(0, self.proposal_std, size=d)
            log_p_proposal = log_posterior(theta_proposal)
            
            # Acceptance ratio (symmetric proposal)
            log_alpha = log_p_proposal - log_p
            
            # Accept/reject
            if np.log(rng.uniform()) < log_alpha:
                theta = theta_proposal
                log_p = log_p_proposal
                if step >= self.burn_in:
                    accepted += 1
            
            total_proposals += 1
            
            # Store sample (after burn-in, with thinning)
            if step >= self.burn_in and (step - self.burn_in) % self.thin == 0:
                samples.append(theta.copy())
                log_posts.append(log_p)
        
        samples = np.array(samples)
        diagnostics = {
            "acceptance_rate": accepted / max(1, total_proposals - self.burn_in),
            "log_posterior_trace": np.array(log_posts),
        }
        
        return samples, diagnostics


class GibbsSampler(MCMCSampler):
    """
    Gibbs sampling MCMC.
    
    Special case of Metropolis-Hastings where each parameter is updated
    one at a time from its full conditional distribution:
        θ_j^(t+1) ~ p(θ_j | θ_1^(t+1), ..., θ_{j-1}^(t+1), θ_{j+1}^(t), ..., θ_d^(t))
    
    Every proposal is accepted (acceptance rate = 100%) because we sample
    directly from the conditional.
    
    Most efficient when full conditionals are available in closed form.
    
    Common use case: Gaussian mixture models, LDA, Ising model.
    
    Args:
        burn_in: Number of initial samples to discard.
        thin: Keep every thin-th sample.
    """

    def __init__(self, burn_in: int = 1000, thin: int = 1):
        self.burn_in = burn_in
        self.thin = thin

    def sample(
        self,
        log_posterior: Callable[[Array], float],
        n_samples: int,
        initial: Optional[Array] = None,
        conditional_samplers: Optional[List[Callable]] = None,
    ) -> Tuple[Array, Dict[str, Array]]:
        """
        Run Gibbs sampling.
        
        Args:
            log_posterior: Log posterior (used for diagnostics only).
            n_samples: Number of samples.
            initial: Initial parameter values.
            conditional_samplers: List of functions, each takes (current_theta, j)
                                  and returns a sample for parameter j.
        
        Returns:
            Tuple of (samples, diagnostics).
        """
        if conditional_samplers is None:
            raise ValueError("Gibbs sampling requires conditional_samplers")
        
        d = len(conditional_samplers)
        if initial is None:
            initial = np.zeros(d)
        
        theta = np.array(initial, dtype=np.float64)
        total_steps = self.burn_in + n_samples * self.thin
        samples = []
        
        for step in range(total_steps):
            # Sample each parameter from its full conditional
            for j in range(d):
                theta[j] = conditional_samplers[j](theta, j)
            
            if step >= self.burn_in and (step - self.burn_in) % self.thin == 0:
                samples.append(theta.copy())
        
        return np.array(samples), {}


# ================================================================
# VARIATIONAL INFERENCE
# ================================================================

class VariationalInference:
    """
    Mean-field variational inference.
    
    Approximates the posterior p(θ|D) with a simpler distribution q(θ)
    by minimizing the KL divergence:
        q* = argmin_q KL(q(θ) || p(θ|D))
    
    Equivalently, maximizing the Evidence Lower BOund (ELBO):
        ELBO = E_q[log p(D,θ)] - E_q[log q(θ)]
             = E_q[log p(D|θ)] - KL(q(θ) || p(θ))
    
    Mean-field assumption: q(θ) = prod_j q_j(θ_j)
    (each parameter has its own independent variational distribution)
    
    Coordinate Ascent Variational Inference (CAVI):
        For each j:
            q_j*(θ_j) ∝ exp(E_{-j}[log p(D,θ)])
        where E_{-j} is the expectation over all other q_k, k != j.
    
    For Gaussian variational family:
        q(θ) = N(μ, diag(σ^2))
        Updates: μ_j, σ_j^2 from the expected log joint.
    
    More efficient than MCMC (no sampling needed), but may underestimate
    posterior variance (especially for multimodal posteriors).
    """

    def __init__(
        self,
        n_dim: int,
        n_iterations: int = 1000,
        learning_rate: float = 0.01,
    ):
        self.n_dim = n_dim
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def fit(
        self,
        log_posterior_unnorm: Callable[[Array], float],
        initial_mean: Optional[Array] = None,
        initial_std: float = 1.0,
    ) -> Dict[str, Array]:
        """
        Fit variational distribution using gradient ascent on ELBO.
        
        Uses the reparameterization trick:
            θ = μ + σ * ε,  ε ~ N(0, I)
        
        This makes the ELBO differentiable w.r.t. μ and σ.
        
        Args:
            log_posterior_unnorm: Unnormalized log posterior.
            initial_mean: Initial variational mean.
            initial_std: Initial variational standard deviation.
        
        Returns:
            Dictionary with 'mean', 'std', 'elbo_trace'.
        """
        # Initialize variational parameters
        mu = np.zeros(self.n_dim) if initial_mean is None else initial_mean.copy()
        log_sigma = np.full(self.n_dim, np.log(initial_std))
        
        elbo_trace = []
        rng = np.random.RandomState(42)
        n_samples = 10  # Monte Carlo samples for ELBO estimate
        
        for iteration in range(self.n_iterations):
            # Reparameterization trick
            epsilon = rng.randn(n_samples, self.n_dim)
            sigma = np.exp(log_sigma)
            theta = mu + sigma * epsilon  # (n_samples, n_dim)
            
            # Evaluate log posterior for each sample
            log_posts = np.array([log_posterior_unnorm(t) for t in theta])
            
            # Entropy of q: H(q) = sum_j [0.5 * log(2*pi*e*sigma_j^2)]
            entropy = 0.5 * self.n_dim * (1 + np.log(2 * np.pi)) + np.sum(log_sigma)
            
            # ELBO = E_q[log p(θ|D)] + H(q)
            elbo = np.mean(log_posts) + entropy
            elbo_trace.append(elbo)
            
            # Gradients via reparameterization trick
            sigma_sq = sigma ** 2
            grad_mu = np.mean(log_posts[:, None] * epsilon, axis=0)
            grad_log_sigma = np.mean(log_posts[:, None] * epsilon * sigma, axis=0) + 1.0
            
            # Update
            mu += self.learning_rate * grad_mu
            log_sigma += self.learning_rate * 0.1 * grad_log_sigma
        
        return {
            "mean": mu,
            "std": np.exp(log_sigma),
            "elbo_trace": np.array(elbo_trace),
        }
