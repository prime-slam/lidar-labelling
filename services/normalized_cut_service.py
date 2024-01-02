# Copyright (c) 2023, Sofia Vivdich and Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy import sparse


def cut_cost(W, mask):
    return (np.sum(W) - np.sum(W[mask][:, mask]) - np.sum(W[~mask][:, ~mask])) / 2


def ncut_cost(W, D, cut):
    """Calculating the value of the normalized similarity criterion for current graph cut

    Parameters
    ----------
    W : matrix
        distance matrix, shows the distance between points and the weight of the corresponding connecting edges
    D : matrix
        diagonal matrix obtained by transforming the distance matrix
    cut : array
        current n-cut, whose cost needs to be calculated
    """
    cost = cut_cost(W, cut)
    # Anastasiia: this also can be optimized in the future
    assoc_a = D.todense()[cut].sum()
    assoc_b = D.todense()[~cut].sum()
    return (cost / assoc_a) + (cost / assoc_b)


def get_min_ncut(ev, d, w, num_cuts):
    """Construction of a minimal graph cut based on a normalized similarity criterion

    Parameters
    ----------
    ev : eigenvector
    d : matrix
        diagonal matrix obtained by transforming the distance matrix
    w : matrix
        distance matrix, shows the distance between points and the weight of the corresponding connecting edges
    num_cuts : int
        number of generated graph cuts, among which the minimum will be selected
    """

    mcut = np.inf
    mn = ev.min()
    mx = ev.max()

    # If all values in `ev` are equal, it implies that the graph can't be
    # further sub-divided. In this case the bi-partition is the graph
    # itself and an empty set.
    min_mask = np.zeros_like(ev, dtype=bool)
    if np.allclose(mn, mx):
        return min_mask, mcut

    # Refer Shi & Malik 2001, Section 3.1.3, Page 892
    # Perform evenly spaced n-cuts and determine the optimal one.
    for t in np.linspace(mn, mx, num_cuts, endpoint=False):
        mask = ev > t
        cost = ncut_cost(w, d, mask)
        if cost < mcut:
            min_mask = mask
            mcut = cost

    return min_mask, mcut


def normalized_cut(w, labels, T=0.01, eigenvalues_count=2):
    """Implementation of the GraphCut algorithm for segmentation labels based on a matrix of distances W between them

    Parameters
    ----------
    w : matrix
        distance matrix, shows the distance between points and the weight of the corresponding connecting edges
    labels : array
        objects that will be divided into clusters; cloud points
    T : float
        criterion for stopping recursive calls to the algorithm
    eigenvalues_count : int
        number of eigenvalues that need to be calculated during the algorithm
    """

    W = w + sparse.identity(w.shape[0])

    if W.shape[0] > 2:
        d = np.array(W.sum(axis=0))[0]
        d2 = np.reciprocal(d)
        D = sparse.diags(d)
        D2 = sparse.diags(d2)

        A = D2 * (D - W) * D2

        eigvals, eigvecs = sparse.linalg.eigsh(
            A, eigenvalues_count, sigma=0, which="LM"
        )

        index2 = np.argsort(eigvals)[1]

        ev = eigvecs[:, index2]

        mask, mcut = get_min_ncut(ev, D, w, 10)
        print("mcut = {}".format(mcut))

        if mcut < T:
            labels1 = normalized_cut(
                w[mask][:, mask], labels[mask], T, eigenvalues_count
            )
            labels2 = normalized_cut(
                w[~mask][:, ~mask], labels[~mask], T, eigenvalues_count
            )
            return labels1 + labels2
        else:
            return [labels]
    else:
        return [labels]
