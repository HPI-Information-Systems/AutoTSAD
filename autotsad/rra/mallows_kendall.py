# Copied and modified from:
# https://github.com/mononitogoswami/tsad-model-selection/blob/master/src/tsadams/model_selection/rank_aggregation.py
# Modifications:
# - Removed 'kemeny' and all related functions
# - Removed 'pagerank' as supported metric in influence-calculation
# - Removed 'weights' parameter from all functions
# - Fixed and added type hints

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def merge(left, right):
    """
    This function uses Merge sort algorithm to count the number of
    inversions in a permutation of two parts (left, right).
    Parameters
    ----------
    left: ndarray
        The first part of the permutation
    right: ndarray
        The second part of the permutation
    Returns
    -------
    result: ndarray
        The sorted permutation of the two parts
    count: int
        The number of inversions in these two parts.
    """
    result = []
    count = 0
    i, j = 0, 0
    left_len = len(left)
    while i < left_len and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            count += left_len - i
            j += 1
    result += left[i:]
    result += right[j:]

    return result, count


def mergeSort_rec(lst):
    """
    This function splits recursively lst into sublists until sublist size is 1. Then, it calls the function merge()
    to merge all those sublists to a sorted list and compute the number of inversions used to get that sorted list.
    Finally, it returns the number of inversions in lst.
    Parameters
    ----------
    lst: ndarray
        The permutation
    Returns
    -------
    result: ndarray
        The sorted permutation
    d: int
        The number of inversions.
    """
    lst = list(lst)
    if len(lst) <= 1:
        return lst, 0
    middle = int(len(lst) / 2)
    left, a = mergeSort_rec(lst[:middle])
    right, b = mergeSort_rec(lst[middle:])
    sorted_, c = merge(left, right)
    d = (a + b + c)
    return sorted_, d


def distance(A, B=None):
    """
    This function computes Kendall's-tau distance between two permutations
    using Merge sort algorithm.
    If only one permutation is given, the distance will be computed with the
    identity permutation as the second permutation

    Parameters
    ----------
    A: ndarray
         The first permutation
    B: ndarray, optional
         The second permutation (default is None)
    Returns
    -------
    int
        Kendall's-tau distance between both permutations (equal to the number of inversions in their composition).
    """
    if B is None:
        B = list(range(len(A)))

    A = np.asarray(A).copy()
    B = np.asarray(B).copy()
    n = len(A)

    # check if A contains NaNs
    msk = np.isnan(A)
    indexes = np.array(range(n))[msk]

    if indexes.size:
        A[indexes] = n  #np.nanmax(A)+1

    # check if B contains NaNs
    msk = np.isnan(B)
    indexes = np.array(range(n))[msk]

    if indexes.size:
        B[indexes] = n  #np.nanmax(B)+1

    # print(A,B,n)
    inverse = np.argsort(B)
    compose = A[inverse]
    _, distance = mergeSort_rec(compose)
    return distance


def median(rankings):  # Borda
    """ This function computes the central permutation (consensus ranking) given
    several permutations.
    Parameters
    ----------
    rankings: ndarray
        Matrix of several permutations
    Returns
    -------
    ndarray
        The central permutation of permutations given.
    """
    consensus = np.argsort(  # give the inverse of result --> sigma_0
        np.argsort(  # give the indexes to sort the sum vector --> sigma_0^-1
            rankings.sum(axis=0)  # sum the indexes of all permutations
        ))
    return consensus


def borda_partial(rankings, w, k):
    """This function approximate the consensus ranking of a top-k rankings using Borda algorithm.
    Each nan-ranked item is assumed to have ranking $k$
    Parameters
    ----------
    rankings: ndarray
        The matrix of permutations
    w: float
        weight of each ranking
    k: int
        Length of partial permutations (only top items)
    Returns
    -------
    ndarray
        Consensus ranking.
    """
    a, b = rankings, w
    a, b = np.nan_to_num(rankings, nan=k), w
    aux = a * b
    borda = np.argsort(np.argsort(np.nanmean(aux, axis=0))).astype(float)
    mask = np.isnan(rankings).all(axis=0)
    borda[mask] = np.nan
    return borda


def check_theta_phi(theta, phi):
    """This function automatically converts theta to phi or phi to theta as
    list or float depending on the values and value types given as input.
    Parameters
    ----------
    theta: float or list
        Dispersion parameter theta to convert to phi (can be None)
    phi: float or list
        Dispersion parameter phi to convert to theta (can be None)
    Returns
    -------
    tuple
        tuple containing both theta and phi (of list or float type depending on the input type)
    """
    if not ((phi is None) ^ (theta is None)):
        print("Set valid values for phi or theta")
    if phi is None and type(theta) != list:
        phi = theta_to_phi(theta)
    if theta is None and type(phi) != list:
        theta = phi_to_theta(phi)
    if phi is None and type(theta) == list:
        phi = [theta_to_phi(t) for t in theta]
    if theta is None and type(phi) == list:
        theta = [phi_to_theta(p) for p in phi]
    return np.array(theta), np.array(phi)


def theta_to_phi(theta):
    """ This functions converts theta dispersion parameter into phi
    Parameters
    ----------
    theta: float
        Real dispersion parameter
    Returns
    -------
    float
        phi real dispersion parameter
    """
    return np.exp(-theta)


def phi_to_theta(phi):
    """This functions converts phi dispersion parameter into theta
    Parameters
    ----------
    phi: float
        Real dispersion parameter
    Returns
    -------
    float
        theta real dispersion parameter
    """
    return -np.log(phi)
