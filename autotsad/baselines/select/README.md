# Implementation for baseline SELECT

SELECT is an ensemble approach for anomaly detection that automatically and systematically selects the results from constituent detectors to combine in a fully unsupervised fashion.

The reference implementation of the SELECT approach is provided as Matlab files.
We re-implement the proposed approach for time series anomaly detection in Python because

- Matlab is not open source and not free (Octave is an alternative, but does not provide full Matlab support) and
- the proposed approach is not directly applicable to time series data.

We just implement the required components of SELECT for our use case and do not provide a full implementation.

## Implementation Notes

The following components of SELECT are implemented:

- Consensus strategies
  - Rank-based
    - Inverse Rank
    - Kemeny-Young
    - RRA
  - Score-based
    - Unification Average
    - Unification Maximum
    - Mixture Modeling (MM) Average
    - Mixture Modeling (MM) Maximum
- Selection strategies
  - Vertical Selection
  - Horizontal Selection


Further notes:

- As the base algorithms, we use the same implementations as AutoTSAD.
  The base algorithms in the paper are proposed for event detection and not for time series anomaly detection.
  In addition, they are not part of the contribution of the paper:
  "SELECT is a flexible approach, as such one can easily expand the ensemble with other base detectors." [p. 42:6]
- SELECT considers two different output modes: (1) raw outlierness scores and (2) rank lists (order of data points according to their outlierness).
  We use the point-based anomaly scores of the base algorithms as raw outlierness scores.
  For the rank lists, we use the order of the data points according to their inverse outlierness scores (increasing rank with decreasing score).
  Because our evaluation measures are designed for anomaly scores, we need to transform the final rank lists back to anomaly scores after consensus.
  We use the inverse rank as anomaly score (increasing score with decreasing rank).
