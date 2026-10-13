import json

from pytest_triage import report


def test_page_renders_from_results_and_escapes_names(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    monkeypatch.setattr(report, "SHADOW", tmp_path)
    labels = {
        label: {"f1": 0.5, "precision": 0.5, "recall": 0.5, "support": 3} for label in report.LABELS
    }
    (tmp_path / "labels.json").write_text(
        json.dumps(
            {
                "<script>": {"macro_f1": 0.4, "micro_f1": 0.5, "n": 10, "labels": labels},
            }
        )
    )
    page = report.html_page()
    assert "&lt;script&gt;" in page and "<script>" not in page
    assert "0.40" in page


def test_markdown_tables_sort_best_first(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    (tmp_path / "labels.json").write_text(
        json.dumps(
            {
                "a": {"macro_f1": 0.1, "micro_f1": 0.1, "n": 1, "labels": {}},
                "b": {"macro_f1": 0.9, "micro_f1": 0.9, "n": 1, "labels": {}},
            }
        )
    )
    table = report.markdown_tables()
    assert table.index("| b |") < table.index("| a |")


def test_selective_rows_show_dev_choice_on_both_splits():
    point = {
        "threshold": 0.9,
        "precision": 0.8,
        "precision_ci95": [0.7, 0.9],
        "recall": 0.2,
        "issue_coverage": 0.17,
    }
    data = {phase: {"chosen_on_dev": {"llm": point}} for phase in ("dev", "test")}
    rows = report.selective_rows(data)
    assert [r[0] for r in rows] == ["dev", "test"]
    assert rows[1][3] == "0.80 (0.70-0.90)" and rows[1][5] == "17%"
