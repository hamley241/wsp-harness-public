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

    # Intern suffixes so equal continuation requirements can be propagated.
    suffix_id = {"": 0}
    first = [-1]
    rest = [-1]
    length = [0]

    def intern(s):
        if s not in suffix_id:
            tail = intern(s[1:])
            suffix_id[s] = len(first)
            first.append(int(s[0]))
            rest.append(tail)
            length.append(len(s))
        return suffix_id[s]

    samples = [(intern(s), y) for s, y in data]
    samples.sort(key=lambda p: length[p[0]])
    ns = len(first)

    def infer(n, budget, limit=96):
        transitions = [-1] * (2 * n)
        solutions = []
        seen = set()
        nodes = 0

        def visit(used):
            nonlocal nodes
            if nodes >= budget or len(solutions) >= limit:
                return
            nodes += 1
            requirements = {}
            blocked = {}

            # A state cannot give different answers to the same suffix.
            for suffix, label in samples:
                state = 0
                while True:
                    key = state * ns + suffix
                    old = requirements.get(key)
                    if old is not None:
                        if old != label:
                            return
                        break
                    requirements[key] = label
                    if suffix == 0:
                        break
                    edge = 2 * state + first[suffix]
                    target = transitions[edge]
                    if target < 0:
                        blocked.setdefault(edge, []).append(
                            (rest[suffix], label)
                        )
                        break
                    state = target
                    suffix = rest[suffix]

            if not blocked:
                table = tuple(
                    max(0, t) for t in transitions[:2 * used]
                )
                labels = tuple(
                    requirements.get(state * ns, 0)
                    for state in range(used)
                )
                signature = (table, labels)
                if signature not in seen:
                    seen.add(signature)
                    solutions.append(signature)
                return

            edge = min(
                blocked,
                key=lambda e: (
                    min(length[s] for s, _ in blocked[e]),
                    -len(blocked[e]),
                    e,
                ),
            )
            pending = blocked[edge]
            choices = []

            for target in range(used + int(used < n)):
                local = {}
                score = 0
                compatible = True
                for suffix, label in pending:
                    state = target
                    while True:
                        key = state * ns + suffix
                        old = requirements.get(key)
                        if old is not None:
                            if old != label:
                                compatible = False
                            else:
                                score += 1
                            break
                        old = local.get(key)
                        if old is not None:
                            if old != label:
                                compatible = False
                            break
                        local[key] = label
                        if suffix == 0:
                            break
                        nxt = transitions[2 * state + first[suffix]]
                        if nxt < 0:
                            break
                        state = nxt
                        suffix = rest[suffix]
                    if not compatible:
                        break
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
        candidates = infer(n, 500000 if n <= 4 else 2000000)
        if candidates:
            break
    if not candidates:
        candidates = infer(8, 8000000, limit=1)
    if not candidates:
        raise RuntimeError("No consistent automaton found")

    smallest = min(len(labels) for _, labels in candidates)
    candidates = [
        (table, labels) for table, labels in candidates
        if len(labels) == smallest
    ]

    # Select a central minimum-state explanation, considering both short
    # exhaustive probes and longer uniformly sampled continuations.
    rng = np.random.default_rng(173)
    probes = rng.integers(0, 2, size=(768, 48))
    predictions = []
    for flat, outputs in candidates:
        table = np.asarray(flat, dtype=np.int64).reshape(-1, 2)
        labels = np.asarray(outputs, dtype=np.int64)
        states = np.array([0], dtype=np.int64)
        parts = []
        for _ in range(8):
            states = table[states].reshape(-1)
            parts.append(labels[states])
        states = np.zeros(len(probes), dtype=np.int64)
        for t in range(probes.shape[1]):
            states = table[states, probes[:, t]]
            if t in (11, 23, 47):
                parts.append(labels[states])
        predictions.append(np.concatenate(parts))

    predictions = np.asarray(predictions, dtype=np.float64)
    consensus = predictions.mean(axis=0)
    distances = np.abs(predictions - consensus).mean(axis=1)
    flat, outputs = candidates[int(np.argmin(distances))]
    table = np.asarray(flat, dtype=np.int64).reshape(-1, 2)
    labels = np.asarray(outputs, dtype=np.int64)
    n = len(labels)

    # Compile the DFA into a saturated bipolar recurrent representation.
    # Unit (q,b) is positive exactly when symbol b just entered state q.
    h = 16
    width = 2 * n
    scale = 24.0
    Wx = np.zeros((2, h), dtype=np.float64)
    Wh = np.zeros((h, h), dtype=np.float64)
    bh = np.zeros(h, dtype=np.float64)
    w = np.zeros(h, dtype=np.float64)
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
    correction = (2.0 * labels.sum() - 1.0 - c_value) / code_sum
    w[:width] = np.repeat(labels, 2).astype(np.float64) + correction

    return {
        "Wx": Wx,
        "Wh": Wh,
        "bh": bh,
        "w": w,
        "c": np.array(c_value, dtype=np.float64),
    }
