import numpy as np

from pytest_triage import learned
from pytest_triage.config import LABELS


def issue(number, labels):
    return {"number": number, "labels": list(labels)}


def toy_data(n=60, seed=0):
    """Two separable clusters: 'type: bug' issues near +x, 'type: docs' near -x."""
    rng = np.random.default_rng(seed)
    bugs = rng.normal([2, 0], 0.3, (n, 2))
    docs = rng.normal([-2, 0], 0.3, (n, 2))
    vectors = np.vstack([bugs, docs])
    issues = [issue(i, ["type: bug"]) for i in range(n)] + [
        issue(n + i, ["type: docs"]) for i in range(n)
    ]
    return vectors, issues


def test_label_model_learns_separable_labels():
    vectors, issues = toy_data()
    model = learned.train(issues, vectors, issues, vectors, Cs=(1.0,))
    preds = model.predict(np.array([[2.0, 0.0], [-2.0, 0.0]]))
    assert preds == [{"type: bug"}, {"type: docs"}]


def test_thresholds_are_tuned_per_label_on_the_tuning_set():
    vectors, issues = toy_data()
    model = learned.LabelModel().fit(vectors, issues).tune_thresholds(vectors, issues)
    assert model.thresholds.shape == (len(LABELS),)
    assert all(0.05 <= t <= 0.95 for t in model.thresholds)


def test_combine_gate_boost_and_route():
    probs = {label: 0.0 for label in LABELS} | {"type: bug": 0.95, "topic: fixtures": 0.25}
    llm = {"type: docs", "topic: fixtures"}
    assert learned.combine(llm, probs, "gate:0.3") == set()
    assert learned.combine(llm, probs, "gate:0.2") == {"topic: fixtures"}
    assert learned.combine(llm, probs, "boost:0.9") == llm | {"type: bug"}
    route = {label: "logreg" for label in LABELS} | {"type: docs": "llm"}
    assert learned.combine(llm, probs, "route", route, {"type: bug"}) == {"type: docs", "type: bug"}


def test_gate_keeps_only_labels_the_classifier_backs():
    from pytest_triage.learned import gate

    probs = {"type: bug": 0.95, "topic: config": 0.4}
    assert gate({"type: bug", "topic: config"}, probs, 0.9) == {"type: bug"}
    assert gate({"topic: config"}, probs, 0.9) == set()
