import warnings

import joblib
import numpy as np
import psutil
from sklearn.cluster import KMeans
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from autotsad.system.execution.gaussian_scaler import GaussianScaler
from .consensus import inverse_ranking, robust_rank_aggregation, kemeny_young, mixture_modeling, unification
from .jings_method import unify_em_jings, class_jings


def _weighted_pcorr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float_, order="c")
    b = np.asarray(b, dtype=np.float_, order="c")
    w = np.asarray(w, dtype=np.float_, order="c")
    if a.shape != b.shape or a.shape != w.shape:
        raise ValueError("Shape of a, b and w must be equal")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(w)):
        raise ValueError("a, b and w must be finite")

    def cov(x, y, aweights=None):
        return np.cov(x, y, aweights=aweights)[0, 1]

    d1, d2 = cov(a, a, aweights=w), cov(b, b, aweights=w)
    if d1 == 0 or d2 == 0:
        return 0.

    return cov(a, b, aweights=w) / np.sqrt(d1 * d2)


def _vertical_selection(scores: np.ndarray) -> np.ndarray:
    scores = GaussianScaler().fit_transform(scores)
    indices = np.arange(scores.shape[0])
    selected = np.zeros(scores.shape[0], dtype=np.bool_)
    discarded = np.zeros(scores.shape[0], dtype=np.bool_)

    def _target(x, mask):
        target = np.mean(x[mask, :], axis=0)
        index = np.argsort(target)[::-1]
        weights = 1. / np.arange(1, index.shape[0]+1)[index]
        return target, index, weights

    def _rho(X, y, w):
        rhos = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            rhos[i] = _weighted_pcorr(X[i, :], y, w)
        return rhos

    # pearson correlation of each score to the global target
    target, target_index, weights = _target(scores, np.ones(scores.shape[0], dtype=np.bool_))
    rhos = _rho(scores, target, weights)

    # start selection
    selected[np.argmax(rhos)] = True

    # compute next algorithm to select and check its suitability (or discard)
    while not np.all(np.bitwise_or(selected, discarded)):
        tmp_target, _, tmp_weights = _target(scores, selected)
        mask = ~np.bitwise_or(selected, discarded)
        rhos = _rho(scores[mask], tmp_target, tmp_weights)

        next_idx = indices[mask][np.argmax(rhos)]
        tmp_selected = selected.copy()
        tmp_selected[next_idx] = True

        # FIXME: faster?
        # current_pred = tmp_target
        current_pred = np.mean(scores[selected, :], axis=0)
        next_pred = np.mean(scores[tmp_selected, :], axis=0)
        if _weighted_pcorr(next_pred, target, weights) > _weighted_pcorr(current_pred, target, weights):
            selected[next_idx] = True
        else:
            discarded[next_idx] = True

    return selected


def _horizontal_selection(scores: np.ndarray, ranks: np.ndarray, cut_off, iterations, n_jobs) -> np.ndarray:
    probs = unify_em_jings(scores, cut_off=cut_off, iterations=iterations, n_jobs=n_jobs)
    class_tags = class_jings(probs)

    _, _, beta, ind = robust_rank_aggregation(ranks)

    # majority voting for pseudo ground truth
    class_tags = np.sum(class_tags, axis=1)
    if ranks.shape[0] % 2 == 0:
        class_tags[class_tags < ranks.shape[0] // 2] = 0
    else:
        class_tags[class_tags <= ranks.shape[0] // 2] = 0
    index = (class_tags > 0).nonzero()[0]

    # frequency of occurrence
    freqs = []
    for i in index:
        idx = np.argmin(beta[i, :])
        if idx < ranks.shape[0]:
            freqs.append(ind[i, idx:])
    freqs = np.concatenate(freqs)

    if len(freqs) != 0:
        # starting centroids for k-means
        f, _ = np.histogram(freqs, bins=np.arange(ranks.shape[0] + 1))
        norm_count = f.astype(np.float_) / index.shape[0]
        centroids = np.array([min(norm_count), max(norm_count)]).reshape(2, 1)
        cls = KMeans(n_clusters=2, init=centroids, n_init=1)
        idcs = cls.fit_predict(norm_count.reshape(-1, 1))
        if cls.cluster_centers_[0] > cls.cluster_centers_[1]:
            selected = (idcs == 0).nonzero()[0]
        else:
            selected = (idcs == 1).nonzero()[0]
        return selected
    else:
        # if no majority voting, select all
        return np.ones(ranks.shape[0], dtype=np.bool_)


class Select:
    def __init__(self, cut_off: int = 20, iterations: int = 100, n_jobs: int = 1) -> None:
        self.cut_off = cut_off
        self.iterations = iterations
        self.n_jobs = n_jobs

    def run(self, scores: np.ndarray, ranks: np.ndarray, strategy: str = "horizontal") -> np.ndarray:
        assert np.all(np.isfinite(scores)), "Input scores must be finite"
        assert np.all(np.isfinite(ranks)), "Input ranks must be finite"

        print(f"Phase 1: Selecting base algorithm results ({strategy} strategy) ...")
        if strategy == "vertical":
            selected_idx = _vertical_selection(scores)
        elif strategy == "horizontal":
            selected_idx = _horizontal_selection(scores, ranks, self.cut_off, self.iterations, self.n_jobs)
        else:
            raise ValueError(f"Invalid strategy '{strategy}'")
        print("... done!")

        if selected_idx.sum() == 1:
            print("Only one algorithm selected, skipping remaining phases")
            return ranks[selected_idx, :].ravel()

        print("Phase 2: Applying consensus algorithms ...")
        scores = scores[selected_idx, :]
        ranks = ranks[selected_idx, :]

        # apply consensus algorithms and collect consensus rankings and scores
        jobs = [
            (inverse_ranking, (ranks,)),
            (robust_rank_aggregation, (ranks,)),
            (mixture_modeling, (scores, np.mean)),
            (mixture_modeling, (scores, np.max)),
            (unification, (scores, np.mean)),
            (unification, (scores, np.max)),
        ]

        # skip kemeny-young if memory is not sufficient or if N is too large (quadratic space and time complexity)
        available_bytes = psutil.virtual_memory().available
        # uint16 is required for more than 255 algorithms, otherwise we can use uint8 (1 byte)
        required_bytes = scores.shape[1]**2 * 1 if scores.shape[0] < 255 else 2
        if available_bytes > required_bytes:
            if scores.shape[1] < 100000:
                jobs.append((kemeny_young, (ranks,)))
            else:
                warnings.warn(f"Skipping Kemeny-Young due to large N={scores.shape[1]} > 100000",
                              category=RuntimeWarning)
        else:
            warnings.warn("Skipping Kemeny-Young due to memory constraints "
                          f"({required_bytes/1024**3:.2f}GB required > {available_bytes/1024**3:.2f}GB available)",
                          category=RuntimeWarning)
        with tqdm_joblib(tqdm(desc="Running consensus algorithms", total=len(jobs))):
            results = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(func)(*params) for func, params in jobs
            )
        ra, sa = zip(*results)
        phase2_ranks = np.vstack(ra).astype(np.int_)
        phase2_scores = np.vstack(sa).astype(np.float_)
        assert np.all(np.isfinite(phase2_scores)), "Scores must be finite"
        assert np.all(np.isfinite(phase2_ranks)), "Ranks must be finite"
        print("... done!")

        print(f"Phase 3: Selecting ensemble components ({strategy} strategy) ...")
        if strategy == "vertical":
            selected_idx = _vertical_selection(phase2_scores)
        else:
            selected_idx = _horizontal_selection(phase2_scores, phase2_ranks, self.cut_off, self.iterations, self.n_jobs)
        # final_scores = phase2_scores[selected_idx, :]
        final_ranks = phase2_ranks[selected_idx, :]
        print("... done!")

        print("Phase 4: Using inverse ranking consensus for final ensemble")
        final_ranks, _ = inverse_ranking(final_ranks)
        return final_ranks
