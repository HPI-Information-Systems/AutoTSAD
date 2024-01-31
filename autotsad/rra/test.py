import numpy as np

from .borda import _get_reliability, borda, trimmed_partial_borda


if __name__ == '__main__':
    ranks = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        [2, 1, 3, 5, 4, 6, 7, 8, 9, 0],
        [3, 1, 2, 4, 5, 9, 6, 0, 7, 8],
    ])
    top_k = 3
    print(ranks.shape)
    reliability = _get_reliability(ranks=ranks,
                                   metric='influence',
                                   aggregation_type='borda',
                                   top_k=top_k,
                                   n_neighbors=None)
    top_reliability_metric_rank = ranks[np.argmax(reliability), :]
    print("mim", top_reliability_metric_rank)

    borda_rank = borda(ranks=ranks)
    print("borda", borda_rank)

    trimmed_partial_borda_rank = trimmed_partial_borda(ranks=ranks,
                                                       top_k=None,
                                                       aggregation_type='borda',
                                                       metric='influence')
    print("robust borda", trimmed_partial_borda_rank)
