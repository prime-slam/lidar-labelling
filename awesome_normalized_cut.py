import numpy as np
from scipy import sparse


def cut_cost(W, mask):
    return (np.sum(W) - np.sum(W[mask][:, mask]) - np.sum(W[~mask][:, ~mask])) / 2


def ncut_cost(W, D, cut):
    print("001")
    cost = cut_cost(W, cut)
    print("002")
    assoc_a = D.todense()[cut].sum()  # Anastasiia: this also can be optimized in the future
    print("003")
    assoc_b = D.todense()[~cut].sum()
    print("004")
    return (cost / assoc_a) + (cost / assoc_b)


def get_min_ncut(ev, d, w, num_cuts):
    print("01")
    mcut = np.inf
    print("02")
    mn = ev.min()
    print("03")
    mx = ev.max()
    print("04")

    # If all values in `ev` are equal, it implies that the graph can't be
    # further sub-divided. In this case the bi-partition is the graph
    # itself and an empty set.
    min_mask = np.zeros_like(ev, dtype=bool)
    print("05")
    if np.allclose(mn, mx):
        print("06")
        return min_mask, mcut

    # Refer Shi & Malik 2001, Section 3.1.3, Page 892
    # Perform evenly spaced n-cuts and determine the optimal one.
    print("07")
    for t in np.linspace(mn, mx, num_cuts, endpoint=False):
        print("08")
        mask = ev > t
        print("09")
        cost = ncut_cost(w, d, mask)
        print("010")
        if cost < mcut:
            print("011")
            min_mask = mask
            mcut = cost
    print("012")

    return min_mask, mcut


def normalized_cut(w, labels, T=0.01):
    print("1")
    W = w + sparse.identity(w.shape[0])
    # W = np.array(w, copy=True)
    # W = np.zeros(w.shape)
    # for i in range(w.shape[0]):
    #     for j in range(w.shape[0]):
    #         W[i, j] = w[i, j]
    #         if i == j:
    #             W[i, j] = W[i, j] + 1.0

    print(W.shape)

    if W.shape[0] > 2:
        print("2")
        d = np.array(W.sum(axis=0))[0]
        print("3")
        d2 = np.reciprocal(d)
        print("4")
        D = sparse.diags(d)
        print("5")
        D2 = sparse.diags(d2)
        print("6")

        A = D2 * (D - W) * D2
        print("7")

        eigvals, eigvecs = sparse.linalg.eigsh(A, 2, sigma=0, which='LM')

        print("8")
        index2 = np.argsort(eigvals)[1]
        print("9")

        ev = eigvecs[:, index2]
        print("10")
        mask, mcut = get_min_ncut(ev, D, w, 10)
        print("11")
        print(mcut)
        print("mask={}".format(mask))
        if mcut < T:
            print("12")
            labels1 = normalized_cut(w[mask][:, mask], labels[mask], T)
            print("13")
            labels2 = normalized_cut(w[~mask][:, ~mask], labels[~mask], T)
            print("14")
            return labels1 + labels2
        else:
            print("15")
            return [labels]
    else:
        print("16")
        return [labels]
