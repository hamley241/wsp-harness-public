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
        ("1011110111", 1), ("0101100110", 0), ("1000100011", 0),
        ("10100001101", 1), ("01011000101", 1), ("00001010010", 0),
        ("100011110111", 1), ("011011111110", 0), ("000000000111", 0),
        ("01110011", 1), ("011100111", 1), ("0111010111", 0),
        ("11101011111", 1), ("01110011111", 1), ("111010111111", 1),
    ]
    children, labels = [[-1, -1]], [-1]
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
    incompatible = [0] * size
    for a in range(size - 1, -1, -1):
        for b in range(a + 1, size):
            conflict = labels[a] >= 0 and labels[b] >= 0 and labels[a] != labels[b]
            if not conflict:
                for bit in (0, 1):
                    u, v = children[a][bit], children[b][bit]
                    if u >= 0 and v >= 0 and incompatible[u] & flags[v]:
                        conflict = True
                        break
            if conflict:
                incompatible[a] |= flags[b]
                incompatible[b] |= flags[a]

    distance, leaf_count = [1000000] * size, [0] * size
    for node in range(size - 1, -1, -1):
        if labels[node] >= 0:
            distance[node], leaf_count[node] = 0, 1
        for child in children[node]:
            if child >= 0:
                distance[node] = min(distance[node], distance[child] + 1)
                leaf_count[node] += leaf_count[child]

    def infer(n, budget, limit=192):
        transitions = [-1] * (2 * n)
        solutions, signatures = [], set()
        visits = 0

        def search(used):
            nonlocal visits
            if visits >= budget or len(solutions) >= limit:
                return
            visits += 1
            members, forbidden, outputs = [0] * n, [0] * n, [-1] * n
            blocked, stack = {}, [(0, 0)]
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
                table = tuple(max(0, t) for t in transitions[:2 * used])
                out = tuple(max(0, label) for label in outputs[:used])
                signature = (table, out)
                if signature not in signatures:
                    signatures.add(signature)
                    solutions.append(signature)
                return

            best = None
            for edge, roots in blocked.items():
                bad_mask, nearest, weight = 0, 1000000, 0
                for node in roots:
                    if bad_mask & flags[node]:
                        return
                    bad_mask |= incompatible[node]
                    nearest = min(nearest, distance[node])
                    weight += leaf_count[node]
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
                    best = (key, edge, roots, choices)

            _, edge, roots, choices = best
            ranked = []
            for target in choices:
                score, valid, implied = 0, True, {}
                pending = [(node, target) for node in roots]
                while pending and valid:
                    node, state = pending.pop()
                    label = labels[node]
                    if label >= 0:
                        known = outputs[state]
                        previous = implied.get(state, -1)
                        if ((known >= 0 and known != label)
                                or (previous >= 0 and previous != label)):
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
        candidates = infer(n, 350000 if n <= 4 else 1800000)
        if candidates:
            break
    if not candidates:
        candidates = infer(8, 6000000, limit=64)
    if not candidates:
        raise RuntimeError("No consistent DFA found")
    smallest = min(len(out) for _, out in candidates)
    candidates = [x for x in candidates if len(x[1]) == smallest]

    # Choose the smallest consistent explanation nearest ensemble consensus.
    rng = np.random.default_rng(73129)
    probes = [
        rng.integers(0, 2, size=(256, length), dtype=np.int64)
        for length in (12, 16, 24, 40, 64)
    ]
    predictions = []
    for flat_table, out in candidates:
        table = np.asarray(flat_table, dtype=np.int64).reshape(-1, 2)
        out = np.asarray(out, dtype=np.int64)
        states, parts = np.array([0], dtype=np.int64), []
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

    # Compile the DFA: two saturated units per state, gated by input.
    n, h, scale = len(out), 16, 20.0
    width = 2 * n
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
    result = {
        "Wx": Wx, "Wh": Wh, "bh": bh, "w": w,
        "c": np.array(c_value, dtype=np.float64),
    }
    for text, label in data:
        state = np.zeros(h, dtype=np.float64)
        for ch in text:
            state = np.tanh(Wx[int(ch)] + Wh @ state + bh)
        if int(w @ state + c_value > 0) != label:
            raise RuntimeError("Compiled recurrence failed validation")
    return result
