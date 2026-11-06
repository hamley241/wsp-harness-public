import numpy as np

def build_weights() -> dict[str, np.ndarray]:
    # The labels match odd parity of the number of '1' symbols.
    h = 16
    gain = 20.0
    Wx = np.zeros((2, h), dtype=float)
    Wh = np.zeros((h, h), dtype=float)
    bh = np.zeros(h, dtype=float)
    w = np.zeros(h, dtype=float)

    # Units 2*q+a encode (current parity q, latest symbol a).
    # Exactly one is +1; the other three are -1.
    # Unit 4 distinguishes initialization from subsequent steps.
    bh[4] = gain

    for q in range(2):
        for a in range(2):
            j = 2 * q + a
            previous_parity = q ^ a
            Wx[a, j] = 2.0 * gain
            Wh[j, 2 * previous_parity:2 * previous_parity + 2] = gain

            if previous_parity == 0:
                # At initialization, the previous parity is even.
                bh[j] = -gain
            else:
                Wh[j, 4] = 2.0 * gain
                bh[j] = -3.0 * gain

    w[:4] = [-1.0, -1.0, 1.0, 1.0]
    c = np.array(-0.5, dtype=float)  # Reject the empty string.

    return {"Wx": Wx, "Wh": Wh, "bh": bh, "w": w, "c": c}
