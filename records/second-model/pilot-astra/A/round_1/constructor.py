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
        ("01110011", 1), ("011100111", 1),
        ("0111010111", 0), ("11101011111", 1),
        ("01110011111", 1), ("111010111111", 1),
    ]

    # Prefix-tree acceptor, retaining unlabeled intermediate prefixes.
    children = [[-1, -1]]
    labels = [-1]
    for text, label in data:
        node = 0
        for ch in text:
            bit = int(ch)
            nxt = children[node][bit]
            if nxt < 0:
                nxt = len(children)
                children[node][bit] = nxt
                children.append([-1, -1])
                labels.append(-1)
            node = nxt
        labels[node] = label

    size = len(children)
    flags = [1 << i for i in range(size)]

    # Two prefixes cannot share a DFA state if an observed common suffix
    # gives them opposite labels. Propagate this information backwards.
    incompatible = [0] * size
    for a in range(size - 1, -1, -1):
        for b in range(a + 1, size):
            conflict = (
                labels[a] >= 0 and labels[b] >= 0
                and labels[a] != labels[b]
            )
            if not conflict:
                for bit in (0, 1):
                    u, v = children[a][bit], children[b][bit]
                    if u >= 0 and v >= 0 and incompatible[u] & flags[v]:
                        conflict = True
                        break
            if conflict:
                incompatible[a] |= flags[b]
                incompatible[b] |= flags[a]

    distance = [1000000] * size
    leaf_count = [0] * size
    for node in range(size - 1, -1, -1):
        if labels[node] >= 0:
            distance[node] = 0
            leaf_count[node] = 1
        for child in children[node]:
            if child >= 0:
                distance[node] = min(distance[node], distance[child] + 1)
                leaf_count[node] += leaf_count[child]

    def infer(n, budget, limit=192):
        transitions = [-1] * (2 * n)
        solutions = []
        signatures = set()
        visits = 0

        def search(used):
            nonlocal visits
            if visits >= budget or len(solutions) >= limit:
                return
            visits += 1

            members = [0] * n
            forbidden = [0] * n
            outputs = [-1] * n
            blocked = {}
            stack = [(0, 0)]

            while stack:
                node, state = stack.pop()
                if forbidden[state] & flags[node]:
                    return
                members[state] |= flags[node]
                forbidden[state] |= incompatible[node]
                if labels[node] >= 0:
                    outputs[state] = labels[node]

                for bit in (0, 1):
                    child = children[node][bit]
                    if child < 0:
                        continue
                    edge = 2 * state + bit
                    target = transitions[edge]
                    if target < 0:
                        blocked.setdefault(edge, []).append(child)
                    else:
                        stack.append((child, target))

            if not blocked:
                table = tuple(
                    max(0, target) for target in transitions[:2 * used]
                )
                out = tuple(max(0, label) for label in outputs[:used])
                signature = (table, out)
                if signature not in signatures:
                    signatures.add(signature)
                    solutions.append(signature)
                return

            best = None
            for edge, roots in blocked.items():
                root_mask = 0
                bad_mask = 0
                nearest = 1000000
                weight = 0
                root_label = -1
                for node in roots:
                    if bad_mask & flags[node]:
                        return
                    root_mask |= flags[node]
                    bad_mask |= incompatible[node]
                    nearest = min(nearest, distance[node])
                    weight += leaf_count[node]
                    if labels[node] >= 0:
                        root_label = labels[node]

                choices = [
                    state for state in range(used)
                    if not (bad_mask & members[state])
                ]
                if used < n:
                    choices.append(used)
                if not choices:
                    return

                key = (len(choices), nearest, -weight, edge)
                if best is None or key < best[0]:
                    best = (key, edge, roots, choices, root_label)

            _, edge, roots, choices, root_label = best

            # Prefer existing states that already explain observed suffixes.
            ranked = []
            for target in choices:
                score = 0
                valid = True
                implied = {}
                pending = [(node, target) for node in roots]
                while pending and valid:
                    node, state = pending.pop()
                    label = labels[node]
                    if label >= 0:
                        known = outputs[state]
                        previous = implied.get(state, -1)
                        if (known >= 0 and known != label) or (
                            previous >= 0 and previous != label
                        ):
                            valid = False
                            break
                        implied[state] = label
                        score += 1 + int(known == label)
                    for bit in (0, 1):
                        child = children[node][bit]
                        if child >= 0:
                            nxt = transitions[2 * state + bit]
                            if nxt >= 0:
                                pending.append((child, nxt))
                if valid:
                    ranked.append((-score, target == used, target))

            ranked.sort()
            for _, _, target in ranked:
                transitions[edge] = target
                search(max(used, target + 1))
                transitions[edge] = -1
                if visits >= budget or len(solutions) >= limit:
                    break

        search(1)
        return solutions

    candidates = []
    for n in range(2, 9):
        budget = 350000 if n <= 4 else 1800000
        candidates = infer(n, budget)
        if candidates:
            break
    if not candidates:
        candidates = infer(8, 6000000, limit=64)
    if not candidates:
        raise RuntimeError("No consistent DFA found")

    smallest = min(len(out) for _, out in candidates)
    candidates = [
        candidate for candidate in candidates
        if len(candidate[1]) == smallest
    ]

    # Select a representative of the smallest consistent explanations,
    # comparing both short strings and substantially longer continuations.
    rng = np.random.default_rng(73129)
    probes = [
        rng.integers(0, 2, size=(256, length), dtype=np.int64)
        for length in (12, 16, 24, 40, 64)
    ]
    predictions = []
    for flat_table, out in candidates:
        table = np.asarray(flat_table, dtype=np.int64).reshape(-1, 2)
        out = np.asarray(out, dtype=np.int64)
        states = np.array([0], dtype=np.int64)
        parts = []
        for _ in range(10):
            states = table[states].reshape(-1)
            parts.append(out[states])
        for bits in probes:
            states = np.zeros(len(bits), dtype=np.int64)
            for column in bits.T:
                states = table[states, column]
            parts.append(out[states])
        predictions.append(np.concatenate(parts))

    predictions = np.asarray(predictions, dtype=np.float64)
    consensus = predictions.mean(axis=0)
    disagreement = np.abs(predictions - consensus).mean(axis=1)
    flat_table, out = candidates[int(np.argmin(disagreement))]
    table = np.asarray(flat_table, dtype=np.int64).reshape(-1, 2)
    out = np.asarray(out, dtype=np.int64)
    n = len(out)

    # Compile the DFA into a saturated tanh recurrence. A state uses two
    # units, distinguished by the last input symbol; one unit is positive.
    h = 16
    width = 2 * n
    scale = 20.0
    Wx = np.zeros((2, h), dtype=np.float64)
    Wh = np.zeros((h, h), dtype=np.float64)
    bh = np.zeros(h, dtype=np.float64)
    w = np.zeros(h, dtype=np.float64)
    code_sum = 2.0 - width

    for target in range(n):
        for bit in (0, 1):
            row = 2 * target + bit
            predecessors = np.flatnonzero(table[:, bit] == target)
            initial = float(table[0, bit] == target)
            correction = (len(predecessors) - initial) / code_sum
            Wh[row, :width] = scale * correction
            for source in predecessors:
                Wh[row, 2 * source:2 * source + 2] += 0.5 * scale
            bh[row] = scale * (initial - 0.5)
            Wx[1 - bit, row] = -2.0 * scale

    c_value = 1.0 if out[0] else -1.0
    correction = (2.0 * out.sum() - 1.0 - c_value) / code_sum
    w[:width] = np.repeat(out, 2).astype(np.float64) + correction

    return {
        "Wx": Wx,
        "Wh": Wh,
        "bh": bh,
        "w": w,
        "c": np.array(c_value, dtype=np.float64),
    }
