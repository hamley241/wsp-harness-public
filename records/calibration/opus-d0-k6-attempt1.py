import numpy as np
import sys

def build_weights():
    data = """1 1
0 0
00 1
01 1
11 1
10 0
001 1
101 1
010 1
111 1
100 1
011 0
000 0
110 0
1100 1
1111 1
1011 1
0011 1
0101 1
1101 1
1001 1
0110 0
0111 0
0010 0
1110 0
1000 0
1010 0
0100 0
10101 1
10100 1
00101 1
00010 1
10010 0
00110 0
11110 0
00011 0
00000 0
01010 0
10110 0
11011 0
111100 1
000000 1
100111 1
111011 1
111000 0
000010 0
100010 0
110010 0
0110101 1
0001100 1
1000011 1
0010100 1
0111110 0
1001000 0
0110111 0
0011000 0
01111100 1
10010001 1
01101001 1
01100100 1
00010000 0
10001011 0
11001000 0
01000000 0
000100111 1
001100010 1
111011100 1
110010100 1
011010010 0
111010110 0
010110010 0
110111110 0
0110100011 1
1101100111 1
0001010101 1
1101010011 1
0100101010 0
1001011000 0
1110111011 0
0111001110 0
00001000101 1
01111001011 1
00101100101 1
11110001111 1
01101111011 0
01001111110 0
11101011011 0
11111011110 0
001011011001 1
101100101000 1
010110101100 1
100101101001 1
000111111110 0
011101110111 0
010101001110 0
110100001110 0"""
    samples = []
    for line in data.strip().split("\n"):
        s, l = line.split()
        samples.append((s, int(l)))

    # APTA
    children = [[-1, -1]]
    lab = [-1]
    par = [-1]
    sym = [-1]
    for s, l in samples:
        node = 0
        for ch in s:
            a = int(ch)
            if children[node][a] < 0:
                children.append([-1, -1]); lab.append(-1)
                par.append(node); sym.append(a)
                children[node][a] = len(children) - 1
            node = children[node][a]
        lab[node] = l
    order = []
    queue = [0]
    qi = 0
    while qi < len(queue):
        nd = queue[qi]; qi += 1
        for a in (0, 1):
            c = children[nd][a]
            if c >= 0:
                order.append(c); queue.append(c)
    N = len(order)
    sys.setrecursionlimit(20000)

    def search(n):
        delta = [-1] * (2 * n)
        slab = [-1] * n
        col = [-1] * len(children)
        col[0] = 0
        slab[0] = lab[0]

        def rec(i, k):
            if i == N:
                return True
            node = order[i]
            q = col[par[node]]
            key = q * 2 + sym[node]
            if delta[key] >= 0:
                cands = [delta[key]]; fresh = False
            else:
                cands = range(min(k + 1, n)); fresh = True
            lb = lab[node]
            for r in cands:
                sl = slab[r]
                if lb >= 0 and sl >= 0 and lb != sl:
                    continue
                setl = lb >= 0 and sl < 0
                if setl:
                    slab[r] = lb
                if fresh:
                    delta[key] = r
                col[node] = r
                if rec(i + 1, max(k, r + 1)):
                    return True
                col[node] = -1
                if fresh:
                    delta[key] = -1
                if setl:
                    slab[r] = -1
            return False

        if rec(0, 1):
            return delta, slab
        return None

    result = None
    for n in range(1, 8):
        result = search(n)
        if result is not None:
            break

    h = 16
    Wx = np.zeros((2, h)); Wh = np.zeros((h, h)); bh = np.zeros(h)
    w = np.zeros(h); c = np.array(0.0)
    if result is None:
        return {"Wx": Wx, "Wh": Wh, "bh": bh, "w": w, "c": c}
    delta, slab = result
    delta = [d if d >= 0 else 0 for d in delta]
    K = 10.0
    s_idx = 2 * n
    bh[s_idx] = K
    nacc = 0
    for qp in range(n):
        for a in (0, 1):
            u = qp * 2 + a
            rel = [q * 2 + b for q in range(n) for b in (0, 1) if delta[q * 2 + a] == qp]
            m = 1.0 if delta[0 * 2 + a] == qp else 0.0
            Wx[a][u] = K
            bh[u] = K * (m - 1.5)
            Wh[u, s_idx] = K * (-m + 0.5 * len(rel))
            for j in rel:
                Wh[u, j] += K / 2
            if slab[qp] == 1:
                w[u] = 1.0
                nacc += 1
    c = np.array(float(nacc - 1))
    return {"Wx": Wx, "Wh": Wh, "bh": bh, "w": w, "c": c}
