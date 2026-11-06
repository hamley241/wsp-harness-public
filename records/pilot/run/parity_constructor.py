import numpy as np

def build_weights():
    h = 16
    B = 10.0
    Wx = np.zeros((2, h))
    Wh = np.zeros((h, h))
    bh = np.zeros(h)
    w = np.zeros(h)
    c = np.array(0.0)

    # states: 0 = even (start), 1 = odd (accept)
    def delta(q, x):
        return q ^ x
    q0 = 0
    def uidx(q, x):
        return q * 2 + x
    z = 4  # start-detector unit
    bh[z] = -B

    for qn in range(2):
        for x in range(2):
            u = uidx(qn, x)
            Wx[x, u] = B
            preds = [q for q in range(2) if delta(q, x) == qn]
            S = [uidx(q, xx) for q in preds for xx in range(2)]
            for j in S:
                Wh[u, j] += B / 2
            bh[u] = B / 2 * len(S) - 1.5 * B
            if q0 not in preds:
                bh[u] += -B
                Wh[u, z] += -B
            w[u] = 1.0 if qn == 1 else -1.0
    return {"Wx": Wx, "Wh": Wh, "bh": bh, "w": w, "c": c}
