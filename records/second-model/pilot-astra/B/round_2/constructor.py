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
        ("000000000111", 0), ("11011010110", 0),
        ("110110", 0), ("000110", 0),
        ("0110111", 0), ("0110000110", 0),
    ]

    # Search for a small consistent DFA, including every counterexample.
    suffixes = {""}
    for text, _ in data:
        suffixes.update(text[i:] for i in range(len(text)))
    suffixes = sorted(suffixes, key=lambda s: (len(s), s))
    index = {s: i for i, s in enumerate(suffixes)}
    first = [-1 if not s else int(s[0]) for s in suffixes]
    tail = [0 if not s else index[s[1:]] for s in suffixes]
    lengths = [len(s) for s in suffixes]
    samples = [
        (index[s], label)
        for s, label in sorted(data, key=lambda item: (len(item[0]), item[0]))
    ]

    def infer(n, budget, limit=512):
        transitions = [-1] * (2 * n)
        solutions = []
        seen = set()
        nodes = 0

        def visit(used):
            nonlocal nodes
            if nodes >= budget or len(solutions) >= limit:
                return
            nodes += 1
            demands = [{} for _ in range(n)]
            blocked = {}

            for initial_suffix, label in samples:
                state = 0
                suffix = initial_suffix
                while True:
                    known = demands[state].get(suffix, -1)
                    if known >= 0 and known != label:
                        return
                    demands[state][suffix] = label
                    if suffix == 0:
                        break

                    edge = 2 * state + first[suffix]
                    suffix = tail[suffix]
                    target = transitions[edge]
                    if target < 0:
                        requirements = blocked.setdefault(edge, {})
                        known = requirements.get(suffix, -1)
                        if known >= 0 and known != label:
                            return
                        requirements[suffix] = label
                        break
                    state = target

            if not blocked:
                table = tuple(max(0, t) for t in transitions[:2 * used])
                labels = tuple(demands[q].get(0, 0) for q in range(used))
                signature = (table, labels)
                if signature not in seen:
                    seen.add(signature)
                    solutions.append(signature)
                return

            edge = min(
                blocked,
                key=lambda e: (
                    min(lengths[s] for s in blocked[e]),
                    -len(blocked[e]),
                    e,
                ),
            )
            requirements = tuple(blocked[edge].items())
            choices = []

            for target in range(min(n, used + 1)):
                transitions[edge] = target
                implied = {}
                compatible = True
                score = 0

                for initial_suffix, label in requirements:
                    state = target
                    suffix = initial_suffix
                    while True:
                        known = demands[state].get(suffix, -1)
                        if known >= 0:
                            if known != label:
                                compatible = False
                                break
                            score += 1

                        key = (state, suffix)
                        known = implied.get(key, -1)
                        if known >= 0 and known != label:
                            compatible = False
                            break
                        implied[key] = label

                        if suffix == 0:
                            score += 2
                            break
                        nxt = transitions[2 * state + first[suffix]]
                        if nxt < 0:
                            break
                        state = nxt
                        suffix = tail[suffix]

                    if not compatible:
                        break

                transitions[edge] = -1
                if compatible:
                    choices.append((-score, target == used, target))

            choices.sort()
            for _, _, target in choices:
                transitions[edge] = target
                visit(max(used, target + 1))
                transitions[edge] = -1
                if nodes >= budget or len(solutions) >= limit:
                    break

        visit(1)
        return solutions

    candidates = []
    for n in range(2, 9):
        budget = 1000000 if n <= 4 else 2000000
        candidates = infer(n, budget)
        if candidates:
            break
    if not candidates:
        candidates = infer(8, 8000000, limit=1)
    if not candidates:
        raise RuntimeError("No consistent DFA found")

    smallest = min(len(labels) for _, labels in candidates)
    candidates = [x for x in candidates if len(x[1]) == smallest]

    # Remove duplicate languages on a broad, deterministic probe set.
    # Then select a consensus representative, not the first search result.
    rng = np.random.default_rng(72419)
    bits = rng.integers(0, 2, size=(96, 1024))
    predictions = []
    unique_candidates = []
    signatures = set()

    for flat_table, labels in candidates:
        table = np.asarray(flat_table, dtype=np.int64).reshape(-1, 2)
        output = np.asarray(labels, dtype=np.int8)
        states = np.array([0], dtype=np.int64)
        parts = []
        for depth in range(1, 11):
            states = table[states].reshape(-1)
            if depth >= 5:
                parts.append(output[states])

        states = np.zeros(bits.shape[1], dtype=np.int64)
        for depth, row in enumerate(bits, 1):
            states = table[states, row]
            if depth in (16, 24, 48, 96):
                parts.append(output[states])

        prediction = np.concatenate(parts)
        signature = prediction.tobytes()
        if signature not in signatures:
            signatures.add(signature)
            unique_candidates.append((flat_table, labels))
            predictions.append(prediction)

    predictions = np.asarray(predictions, dtype=np.float64)
    consensus = predictions.mean(axis=0)
    distances = np.abs(predictions - consensus).mean(axis=1)
    flat_table, labels = unique_candidates[int(np.argmin(distances))]
    table = np.asarray(flat_table, dtype=np.int64).reshape(-1, 2)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)

    for text, expected in data:
        state = 0
        for bit in text:
            state = table[state, int(bit)]
        if labels[state] != expected:
            raise RuntimeError("Inconsistent DFA")

    # Compile the DFA into a saturated bipolar state code. Exactly one
    # of the 2*n active coordinates is positive after each input symbol.
    h = 16
    width = 2 * n
    scale = 20.0
    code_sum = 2.0 - width

    Wx = np.zeros((2, h), dtype=np.float64)
    Wh = np.zeros((h, h), dtype=np.float64)
    bh = np.zeros(h, dtype=np.float64)
    w = np.zeros(h, dtype=np.float64)

    for target in range(n):
        for bit in range(2):
            row = 2 * target + bit
            predecessors = np.flatnonzero(table[:, bit] == target)
            initial = float(table[0, bit] == target)

            # This correction handles both the zero initial vector and
            # the constant-sum bipolar vectors reached subsequently.
            correction = (len(predecessors) - initial) / code_sum
            Wh[row, :width] = scale * correction
            for source in predecessors:
                Wh[row, 2 * source:2 * source + 2] += 0.5 * scale

            bh[row] = scale * (initial - 0.5)
            Wx[1 - bit, row] = -2.0 * scale

    c_value = 1.0 if labels[0] else -1.0
    correction = (2.0 * labels.sum() - 1.0 - c_value) / code_sum
    w[:width] = np.repeat(labels, 2).astype(np.float64) + correction

    result = {
        "Wx": Wx,
        "Wh": Wh,
        "bh": bh,
        "w": w,
        "c": np.array(c_value, dtype=np.float64),
    }

    for text, expected in data:
        state = np.zeros(h, dtype=np.float64)
        for bit in text:
            state = np.tanh(Wx[int(bit)] + Wh @ state + bh)
        if int(w @ state + c_value > 0.0) != expected:
            raise RuntimeError("Numerical compilation failed")

    return result
