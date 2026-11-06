import numpy as np

def build_weights() -> dict[str, np.ndarray]:
    data = [
        ("1", 1), ("0", 0), ("00", 1), ("10", 0),
        ("111", 1), ("000", 0), ("0101", 1), ("1010", 0),
        ("10000", 1), ("01101", 1), ("01010", 0),
        ("110000", 1), ("110010", 0), ("011100", 0),
        ("1111010", 1), ("1111101", 1), ("0111010", 0),
        ("01100101", 1), ("11010010", 0), ("01000110", 0),
        ("111110111", 1), ("110110101", 1), ("111010111", 0),
        ("1011110111", 1), ("0101100110", 0),
        ("1000100011", 0), ("10100001101", 1),
        ("01011000101", 1), ("00001010010", 0),
        ("100011110111", 1), ("011011111110", 0),
        ("000000000111", 0),
    ]
    samples = [(tuple(map(int, s)), y) for s, y in data]
    samples.sort(key=lambda x: len(x[0]))

    # Infer a small deterministic automaton, with canonical state numbering.
    def infer(n, budget, limit=40):
        transitions = [-1] * (2 * n)
        solutions = []
        seen = set()
        nodes = 0

        def visit(used, outputs):
            nonlocal nodes
            if nodes >= budget or len(solutions) >= limit:
                return
            nodes += 1
            outputs = outputs.copy()
            blocked = {}

            for bits, label in samples:
                state = 0
                for pos, bit in enumerate(bits):
                    edge = 2 * state + bit
                    target = transitions[edge]
                    if target < 0:
                        blocked.setdefault(edge, []).append(
                            (bits[pos + 1:], label)
                        )
                        break
                    state = target
                else:
                    if outputs[state] >= 0 and outputs[state] != label:
                        return
                    outputs[state] = label

            if not blocked:
                table = tuple(
                    t if t >= 0 else 0
                    for t in transitions[:2 * used]
                )
                labels = tuple(max(0, y) for y in outputs[:used])
                signature = (table, labels)
                if signature not in seen:
                    seen.add(signature)
                    solutions.append(signature)
                return

            edge = min(
                blocked,
                key=lambda e: (
                    min(len(s) for s, _ in blocked[e]),
                    -len(blocked[e]),
                    e,
                ),
            )
            requirements = blocked[edge]
            choices = []

            for target in range(used + int(used < n)):
                compatible = True
                score = 0
                implied = {}
                for suffix, label in requirements:
                    state = target
                    for bit in suffix:
                        nxt = transitions[2 * state + bit]
                        if nxt < 0:
                            break
                        state = nxt
                    else:
                        known = outputs[state]
                        if known >= 0 and known != label:
                            compatible = False
                            break
                        if state in implied and implied[state] != label:
                            compatible = False
                            break
                        implied[state] = label
                        score += 1 + int(known == label)
                if compatible:
                    choices.append((-score, target == used, target))

            choices.sort()
            for _, _, target in choices:
                transitions[edge] = target
                visit(max(used, target + 1), outputs)
                transitions[edge] = -1
                if nodes >= budget or len(solutions) >= limit:
                    break

        visit(1, [-1] * n)
        return solutions

    candidates = []
    for n in range(2, 9):
        budget = 150000 if n <= 4 else 800000
        candidates = infer(n, budget)
        if candidates:
            break

    if not candidates:
        candidates = infer(8, 6000000, limit=1)
    if not candidates:
        raise RuntimeError("Automaton search exhausted")

    # Among equally small consistent explanations, choose a prediction medoid.
    smallest = min(len(labels) for _, labels in candidates)
    candidates = [
        candidate for candidate in candidates
        if len(candidate[1]) == smallest
    ]
    predictions = []
    for table, labels in candidates:
        table = np.asarray(table, dtype=np.int64).reshape(-1, 2)
        labels = np.asarray(labels, dtype=np.int64)
        states = np.array([0], dtype=np.int64)
        parts = []
        for _ in range(10):
            states = table[states].reshape(-1)
            parts.append(labels[states])
        predictions.append(np.concatenate(parts))
    predictions = np.asarray(predictions, dtype=np.float64)
    majority = predictions.mean(axis=0)
    distances = np.abs(predictions - majority).mean(axis=1)
    table, labels = candidates[int(np.argmin(distances))]
    table = np.asarray(table, dtype=np.int64).reshape(-1, 2)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)

    # Two saturated units per DFA state: one for each incoming symbol.
    # Exactly one unit is positive after each nonempty prefix.
    h = 16
    width = 2 * n
    scale = 20.0
    Wx = np.zeros((2, h), dtype=np.float64)
    Wh = np.zeros((h, h), dtype=np.float64)
    bh = np.zeros(h, dtype=np.float64)
    w = np.zeros(h, dtype=np.float64)

    # Sum of the bipolar code is constant after the first step and zero
    # initially. This supplies initialization without an extra hidden unit.
    code_sum = 2.0 - width
    for target in range(n):
        for bit in range(2):
            row = 2 * target + bit
            predecessors = np.flatnonzero(table[:, bit] == target)
            initial = float(table[0, bit] == target)
            correction = (len(predecessors) - initial) / code_sum
            Wh[row, :width] = scale * correction
            for source in predecessors:
                Wh[row, 2 * source:2 * source + 2] += 0.5 * scale
            bh[row] = scale * (initial - 0.5)
            Wx[1 - bit, row] = -2.0 * scale

    c_value = 1.0 if labels[0] else -1.0
    accepting_units = np.repeat(labels, 2).astype(np.float64)
    correction = (2.0 * labels.sum() - 1.0 - c_value) / code_sum
    w[:width] = accepting_units + correction

    return {
        "Wx": Wx,
        "Wh": Wh,
        "bh": bh,
        "w": w,
        "c": np.array(c_value, dtype=np.float64),
    }
