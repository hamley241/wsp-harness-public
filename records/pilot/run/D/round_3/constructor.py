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
000000000111	0"""


def _parse():
    out = []
    for line in DATA.strip().split("\n"):
        s, l = line.split("\t")
        out.append((s.strip(), int(l)))
    return out


def _trie(data):
    children = [dict()]
    parent = [-1]
    sym = [-1]
    label = [None]
    for s, lab in data:
        v = 0
        for ch in s:
            a = int(ch)
            if a not in children[v]:
                children.append(dict())
                parent.append(v)
                sym.append(a)
                label.append(None)
                children[v][a] = len(children) - 1
            v = children[v][a]
        label[v] = lab
    order = []
    queue = [0]
    while queue:
        nq = []
        for v in queue:
            order.append(v)
            for a in (0, 1):
                if a in children[v]:
                    nq.append(children[v][a])
        queue = nq
    return parent, sym, label, order


def _find_dfa(tr, n, limit=3000000):
    parent, sym, label, order = tr
    M = len(order)
    assign = [-1] * len(parent)
    assign[0] = 0
    delta = {}
    slab = [None] * n
    scnt = [0] * n
    steps = [0]
    if label[0] is not None:
        slab[0] = label[0]
        scnt[0] = 1

    def setlab(q, lab):
        if lab is None:
            return True
        if slab[q] is None:
            slab[q] = lab
            scnt[q] = 1
            return True
        if slab[q] != lab:
            return False
        scnt[q] += 1
        return True

    def unsetlab(q, lab):
        if lab is None:
            return
        scnt[q] -= 1
        if scnt[q] == 0:
            slab[q] = None

    def rec(idx, k):
        steps[0] += 1
        if steps[0] > limit:
            return None
        if idx == M:
            return k
        v = order[idx]
        if v == 0:
            return rec(idx + 1, k)
        q = assign[parent[v]]
        a = sym[v]
        key = (q, a)
        if key in delta:
            t = delta[key]
            if not setlab(t, label[v]):
                return None
            assign[v] = t
            r = rec(idx + 1, k)
            if r is not None:
                return r
            assign[v] = -1
            unsetlab(t, label[v])
            return None
        for t in range(min(k + 1, n)):
            if not setlab(t, label[v]):
                continue
            delta[key] = t
            assign[v] = t
            r = rec(idx + 1, max(k, t + 1))
            if r is not None:
                return r
            assign[v] = -1
            del delta[key]
            unsetlab(t, label[v])
            if steps[0] > limit:
                return None
        return None

    k = rec(0, 1)
    if k is None:
        return None
    labels = {q: slab[q] for q in range(n) if slab[q] is not None}
    return dict(delta), labels, k


def build_weights():
    h = 16
    data = _parse()
    tr = _trie(data)
    res = None
    for n in range(1, 8):
        res = _find_dfa(tr, n)
        if res is not None:
            break
    if res is None:
        n = 1
        delta, labels, k = {}, {0: 0}, 1
    else:
        delta, labels, k = res
        n = k
    trans = [[0, 0] for _ in range(n)]
    for i in range(n):
        for a in range(2):
            t = delta.get((i, a), i)
            trans[i][a] = t if t < n else i
    acc = [labels.get(i, 0) for i in range(n)]

    K = 4.0
    N = 2 * n
    s_idx = N
    Wx = np.zeros((2, h))
    Wh = np.zeros((h, h))
    bh = np.zeros(h)
    w = np.zeros(h)

    for j in range(n):
        for a in range(2):
            u = 2 * j + a
            P = set()
            for i in range(n):
                if trans[i][a] == j:
                    P.add(2 * i)
                    P.add(2 * i + 1)
            for v in range(N):
                Wh[u, v] = K if v in P else -K
            e = 2 * K if trans[0][a] == j else -2 * K
            bh[u] = -2 * K + e
            Wh[u, s_idx] = K * (N - 2 * len(P)) + e
            Wx[a, u] = 2 * K
            Wx[1 - a, u] = -2 * K
    bh[s_idx] = -20.0

    Fu = 0
    for j in range(n):
        for a in range(2):
            u = 2 * j + a
            if acc[j] == 1:
                w[u] = 1.0
                Fu += 1
            else:
                w[u] = -1.0
    c0 = 1.0 if acc[0] == 1 else -1.0
    w[s_idx] = c0 + N - 2 * Fu
    c = np.array(c0)

    return {"Wx": Wx, "Wh": Wh, "bh": bh, "w": w, "c": c}
