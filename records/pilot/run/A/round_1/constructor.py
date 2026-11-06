import numpy as np

DATA = """1	1
0	0
00	1
10	0
111	1
000	0
0101	1
1010	0
10000	1
01101	1
01010	0
110000	1
110010	0
011100	0
1111010	1
1111101	1
0111010	0
01100101	1
11010010	0
01000110	0
111110111	1
110110101	1
111010111	0
1011110111	1
0101100110	0
1000100011	0
10100001101	1
01011000101	1
00001010010	0
100011110111	1
011011111110	0
000000000111	0
011010	1
010010110	0
01001011010	1
010010110100	0"""


def _parse():
    out = []
    for line in DATA.strip().split("\n"):
        s, l = line.split("\t")
        out.append((s.strip(), int(l)))
    out.sort(key=lambda x: (len(x[0]), x[0]))
    return out


def _find_dfa(data, n):
    delta = {}

    def check():
        labels = {}
        miss = None
        for s, lab in data:
            q = 0
            for ch in s:
                t = delta.get((q, int(ch)))
                if t is None:
                    if miss is None:
                        miss = (q, int(ch))
                    q = None
                    break
                q = t
            if q is not None:
                if labels.setdefault(q, lab) != lab:
                    return False, None, None
        return True, miss, labels

    def rec(k):
        ok, miss, labels = check()
        if not ok:
            return None
        if miss is None:
            return dict(delta), dict(labels), k
        for t in range(min(k + 1, n)):
            delta[miss] = t
            r = rec(max(k, t + 1))
            if r is not None:
                return r
            del delta[miss]
        return None

    return rec(1)


def build_weights():
    h = 16
    data = _parse()
    res = None
    for n in range(1, 8):
        res = _find_dfa(data, n)
        if res is not None:
            break
    if res is None:
        n = 1
        delta, labels = {}, {0: 1}
    else:
        delta, labels, k = res
        n = k
    trans = [[delta.get((i, a), i) for a in range(2)] for i in range(n)]
    acc = [labels.get(i, 0) for i in range(n)]

    K = 10.0
    N = 2 * n
    s = N
    Wx = np.zeros((2, h))
    Wh = np.zeros((h, h))
    bh = np.zeros(h)
    w = np.zeros(h)

    for j in range(n):
        for a in range(2):
            u = 2 * j + a
            P = []
            for i in range(n):
                if trans[i][a] == j:
                    P.append(2 * i)
                    P.append(2 * i + 1)
            for v in P:
                Wh[u, v] = K / 2
            e1 = 1.0 if trans[0][a] == j else 0.0
            bh[u] = K * (e1 - 1.5)
            Wh[u, s] = K * e1 - K * len(P) / 2.0
            Wx[a, u] = K
    bh[s] = -20.0

    sacc = 0.0
    for j in range(n):
        for a in range(2):
            u = 2 * j + a
            if acc[j] == 1:
                w[u] = 0.5
                sacc += 0.5
    acc0 = float(acc[0])
    w[s] = acc0 - sacc
    c = np.array(acc0 - 0.5)

    return {"Wx": Wx, "Wh": Wh, "bh": bh, "w": w, "c": c}
