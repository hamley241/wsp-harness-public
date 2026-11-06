import numpy as np
import random


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
        # Retain the additional labels from the previous constructor.
        ("11011010110", 0), ("110110", 0), ("000110", 0),
        ("0110111", 0), ("0110000110", 0),
    ]

    suffixes = {""}
    for text, _ in data:
        suffixes.update(text[i:] for i in range(len(text)))
    suffixes = sorted(suffixes, key=lambda s: (len(s), s))
    index = {s: i for i, s in enumerate(suffixes)}
    first = [-1 if not s else int(s[0]) for s in suffixes]
    tail = [0 if not s else index[s[1:]] for s in suffixes]
    lengths = [len(s) for s in suffixes]
    samples = [
        (index[s], y)
        for s, y in sorted(data, key=lambda x: (len(x[0]), x[0]))
    ]

    def canonical(flat, labels):
        """Minimize, remove unreachable states, and canonically number."""
        n = len(labels)
        table = [flat[2*q:2*q+2] for q in range(n)]
        groups = list(labels)
        while True:
            mapping = {}
            refined = []
            for q in range(n):
                key = (labels[q], groups[table[q][0]],
                       groups[table[q][1]])
                if key not in mapping:
                    mapping[key] = len(mapping)
                refined.append(mapping[key])
            if refined == groups:
                break
            groups = refined

        representatives = {}
        for q, group in enumerate(groups):
            representatives.setdefault(group, q)
        start = groups[0]
        order = [start]
        numbers = {start: 0}
        result_table = []
        result_labels = []
        for group in order:
            q = representatives[group]
            result_labels.append(labels[q])
            for bit in range(2):
                target = groups[table[q][bit]]
                if target not in numbers:
                    numbers[target] = len(order)
                    order.append(target)
                result_table.append(numbers[target])
        return tuple(result_table), tuple(result_labels)

    def infer(n, budget, limit, seed=None):
        transitions = [-1] * (2*n)
        solutions = {}
        nodes = 0
        stopped = False
        rng = random.Random(seed)

        def visit(used):
            nonlocal nodes, stopped
            if nodes >= budget or len(solutions) >= limit:
                stopped = True
                return
            nodes += 1
            demands = [{} for _ in range(n)]
            blocked = {}

            for initial_suffix, label in samples:
                state, suffix = 0, initial_suffix
                while True:
                    known = demands[state].get(suffix, -1)
                    if known >= 0 and known != label:
                        return
                    demands[state][suffix] = label
                    if suffix == 0:
                        break
                    edge = 2*state + first[suffix]
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
                flat = tuple(max(0, x) for x in transitions[:2*used])
                labels = [demands[q].get(0, -1) for q in range(used)]
                unknown = [q for q, value in enumerate(labels) if value < 0]
                # Do not silently force unconstrained acceptance bits to zero.
                for mask in range(1 << len(unknown)):
                    output = labels[:]
                    for j, q in enumerate(unknown):
                        output[q] = (mask >> j) & 1
                    key = canonical(flat, tuple(output))
                    solutions[key] = None
                    if len(solutions) >= limit:
                        stopped = True
                        break
                return

            ranked_edges = sorted(
                blocked,
                key=lambda e: (
                    min(lengths[s] for s in blocked[e]),
                    -len(blocked[e]), e,
                ),
            )
            edge = ranked_edges[0]
            if seed is not None and len(ranked_edges) > 1:
                shortest = min(lengths[s] for s in blocked[edge])
                eligible = [
                    e for e in ranked_edges[:4]
                    if min(lengths[s] for s in blocked[e]) == shortest
                ]
                edge = rng.choice(eligible)
            requirements = tuple(blocked[edge].items())
            choices = []

            for target in range(min(n, used + 1)):
                transitions[edge] = target
                implied = {}
                compatible = True
                score = 0
                for initial_suffix, label in requirements:
                    state, suffix = target, initial_suffix
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
                        nxt = transitions[2*state + first[suffix]]
                        if nxt < 0:
                            break
                        state, suffix = nxt, tail[suffix]
                    if not compatible:
                        break
                transitions[edge] = -1
                if compatible:
                    jitter = 0.0 if seed is None else rng.uniform(-3, 3)
                    choices.append((-score + jitter, target == used, target))

            choices.sort()
            for _, _, target in choices:
                transitions[edge] = target
                visit(max(used, target + 1))
                transitions[edge] = -1
                if nodes >= budget or len(solutions) >= limit:
                    stopped = True
                    break

        visit(1)
        return list(solutions), not stopped

    # Prefer the smallest consistent automata. Diversified searches reduce
    # dependence on the arbitrary numbering and ordering of the first solution.
    pool = {}
    for n in range(2, 9):
        budget = 400000 if n <= 4 else 1000000
        found, complete = infer(n, budget, 256)
        pool.update((candidate, None) for candidate in found)
        if not complete:
            trials = 5 if found else 3
            for trial in range(trials):
                extra, _ = infer(
                    n, 120000, 96, seed=9137 + 101*n + trial
                )
                pool.update((candidate, None) for candidate in extra)
        if pool:
            break

    if not pool:
        found, _ = infer(8, 8000000, 1, seed=17591)
        pool.update((candidate, None) for candidate in found)
    if not pool:
        raise RuntimeError("Unable to infer a consistent DFA")

    smallest = min(len(labels) for _, labels in pool)
    candidates = sorted(
        candidate for candidate in pool if len(candidate[1]) == smallest
    )

    # Select an actual consistent language, rather than averaging logits.
    # Exact minimization above prevents isomorphic DFAs from getting extra votes.
    rng = np.random.default_rng(72419)
    probes = rng.integers(0, 2, size=(128, 768), dtype=np.int64)
    predictions = []
    for flat, labels in candidates:
        table = np.asarray(flat, dtype=np.int64).reshape(-1, 2)
        output = np.asarray(labels, dtype=np.float64)
        parts = []
        states = np.zeros(1, dtype=np.int64)
        for depth in range(1, 12):
            states = table[states].reshape(-1)
            parts.append(output[states])
        states = np.zeros(probes.shape[1], dtype=np.int64)
        for depth, bits in enumerate(probes, 1):
            states = table[states, bits]
            if depth in (16, 24, 32, 64, 128):
                parts.append(output[states])
        predictions.append(np.concatenate(parts))
    predictions = np.asarray(predictions)
    consensus = predictions.mean(axis=0)
    distances = np.abs(predictions - consensus).mean(axis=1)
    flat, labels = candidates[int(np.argmin(distances))]
    table = np.asarray(flat, dtype=np.int64).reshape(-1, 2)
    labels = np.asarray(labels, dtype=np.int64)

    # Two bipolar coordinates per DFA state, indexed by the latest input.
    # Exactly one coordinate is positive after every nonempty prefix.
    n = len(labels)
    width = 2*n
    scale = 32.0
    code_sum = 2.0 - width
    Wx = np.zeros((2, 16), dtype=np.float64)
    Wh = np.zeros((16, 16), dtype=np.float64)
    bh = np.zeros(16, dtype=np.float64)
    w = np.zeros(16, dtype=np.float64)

    for target in range(n):
        for bit in range(2):
            row = 2*target + bit
            predecessors = np.flatnonzero(table[:, bit] == target)
            initial = float(table[0, bit] == target)
            correction = (len(predecessors) - initial) / code_sum
            Wh[row, :width] = scale * correction
            for source in predecessors:
                Wh[row, 2*source:2*source+2] += 0.5*scale
            bh[row] = scale * (initial - 0.5)
            Wx[1-bit, row] = -2.0*scale

    c_value = 1.0 if labels[0] else -1.0
    correction = (2.0*labels.sum() - 1.0 - c_value) / code_sum
    w[:width] = np.repeat(labels, 2) + correction

    for text, expected in data:
        state = np.zeros(16, dtype=np.float64)
        for bit in text:
            state = np.tanh(Wx[int(bit)] + Wh @ state + bh)
        if int(w @ state + c_value > 0.0) != expected:
            raise RuntimeError("Numerical DFA compilation failed")

    return {
        "Wx": Wx,
        "Wh": Wh,
        "bh": bh,
        "w": w,
        "c": np.array(c_value, dtype=np.float64),
    }
