import numpy as np

def build_weights():
    h = 16
    K = 10.0
    Wx = np.zeros((2, h))
    Wh = np.zeros((h, h))
    bh = np.zeros(h)
    w = np.zeros(h)
    # units: 0=(even,0),1=(even,1),2=(odd,0),3=(odd,1)
    # I_e = (h0+h1)/2 + 1 ; I_o = -(h0+h1)/2
    units = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for j, (q, a) in enumerate(units):
        Wx[a, j] += K
        b = -1.5
        if q == 0:
            Wh[j, 0] += K * 0.5
            Wh[j, 1] += K * 0.5
            b += 1.0
        else:
            Wh[j, 0] += -K * 0.5
            Wh[j, 1] += -K * 0.5
        bh[j] = K * b
    # unused units: keep saturated at -1, no influence
    for j in range(4, h):
        bh[j] = -K
    # odd parity units: 1 (even,1) and 2 (odd,0)
    w[1] = 1.0
    w[2] = 1.0
    w[0] = -1.0
    w[3] = -1.0
    c = np.array(-0.5)
    return {"Wx": Wx, "Wh": Wh, "bh": bh, "w": w, "c": c}
