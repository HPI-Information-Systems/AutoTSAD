from typing import Tuple

import numpy as np
from scipy.special import comb


def pyflagr_robust_rank_aggregation(ranks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from pyflagr.RRA import RRA
    import pandas as pd

    df = pd.DataFrame(1 / ranks.T, columns=["model_" + str(i) for i in range(ranks.shape[0])])
    df = df.unstack().reset_index()
    df = df.rename(columns={"level_0": "Voter", "level_1": "Item Code", 0: "Item Score"})
    df["Item Code"] = df["Item Code"].apply(lambda x: f"E-{x}")
    df.insert(0, "Query", 1)
    df["Algorithm/Dataset"] = "unknown"

    # df = pd.read_csv("testdata.csv", header=None)
    # df.columns = ["Query", "Voter", "Item Code", "Item Score", "Algorithm/Dataset"]
    # df = df[df["Query"] == 1]
    # df["Item Code"] = df["Item Code"].str.replace("E", "E-")
    # # df["Item Code"] = df["Item Code"].str.split("E").str[-1].astype(int)
    #
    # # items = set(df["Item Code"].unique())
    # # dataset = df["Algorithm/Dataset"].unique()[0]
    # # for v in df["Voter"].unique():
    # #     v_items = set(df[df["Voter"] == v]["Item Code"].unique())
    # #     missing_items = items - v_items
    # #     df = pd.concat([df,
    # #                     pd.DataFrame({
    # #                         "Query":  1,
    # #                         "Voter": v,
    # #                         "Item Code": list(sorted(missing_items)),
    # #                         "Item Score": 0,
    # #                         "Algorithm/Dataset": dataset}
    # #                     )],
    # #                    ignore_index=True)
    #
    # print("Voters", len(df["Voter"].unique()))
    # print("Items", len(df["Item Code"].unique()))
    # print("Voter-Items", len((df["Voter"] + df["Item Code"]).unique()))
    # print(df.groupby(["Voter"])[["Item Score"]].count())
    # print(df)

    # df = df.sample(frac=1, ignore_index=True, random_state=None)

    df_final_ranks, _ = RRA().aggregate(input_df=df)
    df_final_ranks["Voter"] = df_final_ranks["Voter"].str.split("-").str[-1].astype(int)
    df_ranks = df_final_ranks.pivot(index="Voter", columns="Query", values="ItemID").sort_index()
    df_scores = df_final_ranks.pivot(index="Voter", columns="Query", values="Score").sort_index()
    df_scores = 1 - df_scores
    return df_ranks.iloc[:, 0].values, df_scores.iloc[:, 0].values


def old_robust_rank_aggregation(ranks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # change from row to column-oriented matrix:
    ranks = ranks.T.astype(np.int_)
    m, n = ranks.shape
    r = np.sort(ranks, axis=0)[::-1]

    ind = np.argsort(r, axis=1)
    beta = np.zeros_like(r)

    for i in np.arange(m):
        for k in np.arange(n):
            beta[i, k] = sum(comb(n, l) * r[i, k]**l * (1 - r[i, k])**(n - l) for l in range(k, n))
    rho = np.min(beta, axis=1)

    final_ranklist = np.argsort(rho) + 1
    score = 1 - rho

    return final_ranklist, score#, beta, ind


def robust_rank_aggregation_impl(ranks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ranks = ranks.T.astype(np.int_)
    m, n = ranks.shape
    norm_i = np.arange(1, m + 1) / m
    r = np.zeros((m, n))

    # Calculate normalized rank vector
    for i in range(n):
        sorted_indices = np.argsort(ranks[:, i])
        r[sorted_indices, i] = norm_i

    ind = np.argsort(r, axis=1)
    beta = np.zeros_like(r)

    # Compute binomial probabilities
    for i in range(m):
        for k in range(n):
            beta[i, k] = _binom_prob(k, n, r[i, k])

    # Compute rho using vectorized np.min
    rho = np.min(beta, axis=1)

    final_ranklist = np.argsort(rho) + 1
    score = 1 - rho

    return final_ranklist, score, beta, ind


def _binom_prob(k: int, n: int, rik: np.ndarray) -> float:
    probs = np.empty(n - k, dtype=np.float_)
    for l in range(k, n):
        probs[l - k] = comb(n, l) * rik**l * (1 - rik)**(n - l)
    return np.sum(probs)
