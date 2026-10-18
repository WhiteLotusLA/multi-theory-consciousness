"""The compact temporal memory learns a repeating sequence: after training,
each pattern is predicted by the one before it, a novel pattern is still
anomalous, and a memory that never learns keeps scoring every step as
anomalous."""

from mtc.neural.htm.htm_core import SDR, TemporalMemory

N_COLUMNS = 40
PATTERNS = {
    "A": frozenset(range(0, 5)),
    "B": frozenset(range(10, 15)),
    "C": frozenset(range(20, 25)),
}
NOVEL = frozenset(range(30, 35))


def _memory() -> TemporalMemory:
    return TemporalMemory(
        n_columns=N_COLUMNS,
        cells_per_column=4,
        activation_threshold=2,
        learning_threshold=1,
    )


def _sdr(columns) -> SDR:
    return SDR(size=N_COLUMNS, active_indices=set(columns))


def _run(memory: TemporalMemory, repeats: int, learn: bool = True):
    anomalies = []
    for _ in range(repeats):
        for name in ("A", "B", "C"):
            _, anomaly = memory.compute(_sdr(PATTERNS[name]), learn=learn)
            anomalies.append(anomaly)
    return anomalies


def test_repeating_sequence_becomes_predicted():
    memory = _memory()
    anomalies = _run(memory, repeats=30)

    assert anomalies[0] == 1.0
    tail = anomalies[-6:]
    assert sum(tail) / len(tail) < 0.2

    # Seeing A makes cells of B's columns predictive for the next step.
    memory.compute(_sdr(PATTERNS["A"]), learn=True)
    predicted_columns = {col for col, _ in memory.predictive_cells}
    assert memory.predictive_cells
    assert predicted_columns <= PATTERNS["B"]
    assert predicted_columns == PATTERNS["B"]


def test_novel_pattern_after_training_is_anomalous():
    memory = _memory()
    _run(memory, repeats=30)

    _, anomaly = memory.compute(_sdr(NOVEL), learn=True)
    assert anomaly == 1.0


def test_no_learning_stays_anomalous():
    memory = _memory()
    anomalies = _run(memory, repeats=10, learn=False)

    assert anomalies == [1.0] * len(anomalies)
    assert memory.predictive_cells == set()
