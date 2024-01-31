import joblib
import numpy as np
from numba import njit

# from scipy.stats import norm

EPSILON = 1e-12


def unify_em_jings(scores: np.ndarray, cut_off: int = 20, iterations: int = 100, n_jobs: int = 1) -> np.ndarray:
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_unify_em)(scores[i, :], cut_off, iterations) for i in range(scores.shape[0])
    )
    result = np.vstack(results).astype(np.float_)
    return result


def class_jings(probabilities: np.ndarray) -> np.ndarray:
    classes = (probabilities > 0.5).astype(np.int_)
    return classes


@njit(nogil=True, parallel=True)
def pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2))) / (sigma * np.sqrt(2 * np.pi))


@njit(nogil=True, parallel=True)
def _unify_em(score: np.ndarray, cut_off: int, iterations: int) -> np.ndarray:
    scores = np.sort(score)[::-1]
    idx = np.argsort(score)[::-1]
    N = len(scores)

    temp_mu = 0.
    temp_sigma = 0.
    temp_lambda = 0.
    temp_alpha = 0.

    # Initialize parameters
    fO = scores[:cut_off]
    fM = scores[cut_off:]

    mu = np.mean(fO)
    sigma = np.sum((fO - mu)**2) / len(fO) + EPSILON
    lambda_ = len(fM) / (np.sum(fM) + EPSILON)
    alpha = len(fO) / N

    t = np.zeros(N, dtype=np.bool_)
    t[:cut_off] = True

    for i in range(iterations):
        # E-Step
        # p = norm.pdf(scores, mu, np.sqrt(sigma)
        p = pdf(scores, mu, np.sqrt(sigma))
        q = lambda_ * np.exp(-lambda_ * scores)

        pr_gaus = alpha * p
        pr_expo = (1 - alpha) * q

        loc = pr_gaus > pr_expo
        t[loc] = True
        t[~loc] = False

        # M-Step
        t_mu = mu
        if np.sum(t) > 0 and np.sum(~t) > 0:
            mu = np.sum(t * scores) / np.sum(t)
            # add EPSILON to avoid divide by zero if variance becomes zero for any random data
            sigma = np.sum(t * (scores - t_mu)**2) / np.sum(t) + EPSILON
            lambda_ = np.sum(~t) / (np.sum(~t * scores) + EPSILON)
            alpha = np.sum(t) / N

        if np.allclose([mu, sigma, lambda_, alpha], [temp_mu, temp_sigma, temp_lambda, temp_alpha]):
            print(f"  (Jing's EM) Converged in {i}<{iterations} iterations")
            break

        temp_mu, temp_sigma, temp_lambda, temp_alpha = mu, sigma, lambda_, alpha

    if i == iterations - 1:
        print(f"  (Jing's EM) Warning: EM algorithm did not converge in {iterations} iterations")

    p = pdf(scores, mu, np.sqrt(sigma))
    q = lambda_ * np.exp(-lambda_ * scores)

    posterior_prob = (alpha * p) / ((alpha * p) + ((1 - alpha) * q))

    # _, idx = np.unique(score, return_index=True)
    # idx = np.argsort(idx)
    pr = np.zeros_like(score)
    pr[idx] = posterior_prob

    return pr
