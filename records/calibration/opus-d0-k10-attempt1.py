import numpy as np

DATA = """0	1
1	0
01	1
11	0
00	0
10	0
011	1
111	1
001	1
000	0
010	0
101	0
110	0
100	0
1101	1
1110	1
0010	1
1011	1
0101	1
0001	1
1010	0
0111	0
1100	0
0100	0
0000	0
0110	0
1111	0
00001	1
10011	1
01111	1
00101	1
00111	1
01011	1
10110	1
00011	1
11011	1
11110	0
00000	0
01001	0
11000	0
010011	1
101001	1
100110	1
000011	1
000010	0
001011	0
001110	0
110010	0
0001111	1
1000011	1
0010100	1
1000101	1
0100001	0
1100010	0
0000100	0
1000001	0
11001101	1
00000011	1
00101101	1
01100101	1
01110111	0
01100100	0
10011011	0
00011100	0
010011111	1
001101111	1
100011111	1
110011011	1
101011110	0
110101000	0
000111100	0
011101000	0
0011110101	1
1100110011	1
0000100110	1
0101100110	1
0000101010	0
0111111111	0
0011101110	0
0110100100	0
11110101101	1
10111111011	1
11010101111	1
01000111101	1
01011011010	0
01010010100	0
10100100000	0
01111100010	0
010111001100	1
011010111101	1
001011101111	1
001100110001	1
000111000010	0
001111110110	0
001011100001	0
111010111000	0"""


def parse():
    out = []
    for line in DATA.strip().split("\n"):
        s, l = line.split("\t")
        out.append((s.strip(), int(l)))
    return out


def build_tree(data):
    children = [dict()]
    label = [None]
    parent = [-1]
    sym = [-1]
    for s, l in data:
        node = 0
        for ch in s:
            a = int(ch)
            if a not in children[node]:
                children.append(dict())
                label.append(None)
                parent.append(node)
                sym.append(a)
                children[node][a] = len(children) - 1
            node = children[node][a]
        label[node] = l
    # BFS order
    order = [0]
    i = 0
    while i < len(order):
        nd = order[i]
        for a in (0, 1):
            if a in children[nd]:
                order.append(children[nd][a])
        i += 1
    return children, label, parent, sym, order


def find_dfa(n, label, parent, sym, order):
    N = len(order)
    state = [0] * len(label)
    delta = {}
    acc = [None] * n
    used = [1]
    if label[0] is not None:
        acc[0] = label[0]

    def solve(k):
        trail = []
        while k < N:
            node = order[k]
            p = state[parent[node]]
            a = sym[node]
            if (p, a) in delta:
                s = delta[(p, a)]
                state[node] = s
                lb = label[node]
                if lb is not None:
                    if acc[s] is None:
                        acc[s] = lb
                        trail.append(s)
                    elif acc[s] != lb:
                        for t in trail:
                            acc[t] = None
                        return False
                k += 1
            else:
                break
        if k == N:
            return True
        node = order[k]
        p = state[parent[node]]
        a = sym[node]
        lb = label[node]
        cands = list(range(used[0]))
        if used[0] < n:
            cands.append(used[0])
        for s in cands:
            if lb is not None and acc[s] is not None and acc[s] != lb:
                continue
            isnew = (s == used[0])
            if isnew:
                used[0] += 1
            setacc = False
            if lb is not None and acc[s] is None:
                acc[s] = lb
                setacc = True
            delta[(p, a)] = s
            state[node] = s
            if solve(k + 1):
                return True
            del delta[(p, a)]
            if setacc:
                acc[s] = None
            if isnew:
                used[0] -= 1
        for t in trail:
            acc[t] = None
        return False

    if solve(1):
        return dict(delta), [x if x is not None else 0 for x in acc], used[0]
    return None


def build_weights():
    h = 16
    data = parse()
    children, label, parent, sym, order = build_tree(data)
    result = None
    for n in range(1, 10):
        r = find_dfa(n, label, parent, sym, order)
        if r is None:
            continue
        delta, acc, nused = r
        # live states: can reach accepting
        live = set(q for q in range(nused) if acc[q] == 1)
        changed = True
        while changed:
            changed = False
            for (p, a), q in delta.items():
                if q in live and p not in live:
                    live.add(p)
                    changed = True
        units = [(p, a) for (p, a), q in delta.items() if p in live and q in live]
        if len(units) + 1 <= h:
            result = (delta, acc, live, units)
            break
    Wx = np.zeros((2, h))
    Wh = np.zeros((h, h))
    bh = np.zeros(h)
    w = np.zeros(h)
    c = np.array(0.0)
    if result is None:
        rng = np.random.RandomState(0)
        return {"Wx": rng.randn(2, h) * 0.1, "Wh": rng.randn(h, h) * 0.1,
                "bh": np.zeros(h), "w": rng.randn(h) * 0.1, "c": np.array(0.0)}
    delta, acc, live, units = result
    A = 20.0
    start = 0
    bh[0] = A  # z unit
    idx = {u: i + 1 for i, u in enumerate(units)}
    for (pp, ap), j in idx.items():
        target_state = pp  # unit (pp, ap): previous state pp, input ap
        S = [idx[(p, a)] for (p, a) in units if delta[(p, a)] == target_state]
        is_start = 1.0 if target_state == start else 0.0
        for i in S:
            Wh[j, i] += A / 2.0
        cj = len(S) / 2.0 - is_start
        Wh[j, 0] = A * cj
        bh[j] = A * is_start - 1.5 * A
        Wx[ap, j] = A
    total = 0.0
    for (p, a), j in idx.items():
        av = 1.0 if acc[delta[(p, a)]] == 1 else 0.0
        w[j] = av / 2.0
        total += av / 2.0
    e = 0.5 if acc[start] == 1 else -0.5
    w[0] = total - 0.5 - e
    c = np.array(e)
    return {"Wx": Wx, "Wh": Wh, "bh": bh, "w": w, "c": c}
