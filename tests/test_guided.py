from pytest_triage import guided


def issue(number, labels=(), body="text"):
    return {"number": number, "title": f"issue {number}", "body": body, "labels": list(labels)}


KINDS = ["type: bug", "type: question", "type: docs", "type: enhancement", "type: proposal"]
TEST = [issue(n, [KINDS[n % 5]]) for n in range(1, 41)] + [issue(99, ["topic: fixtures"])]


def plan(warmup=(1, 2), main=({"number": 3, "label": "type: bug"},)):
    return {
        "warmup": list(warmup),
        "main": list(main),
        "meta": {"tp": 10, "fp": 10},
    }


def session(answers, p, tmp_path, comments=None):
    feed, shown = iter(answers), []
    done = guided.run_session(
        "me",
        p,
        {i["number"]: i for i in TEST},
        lambda n: comments or [],
        ask=lambda _: next(feed),
        show=shown.append,
        directory=tmp_path,
    )
    return done, "\n".join(shown)


def test_warmup_is_ten_single_type_issues_and_reproducible():
    first = guided.draw_warmup(TEST)
    assert len(first) == 10 and len(set(first)) == 10
    assert first == guided.draw_warmup(list(reversed(TEST)))
    assert 99 not in first  # no type label: no right answer to compare with


def test_main_sample_keeps_only_type_labels():
    fps = [{"number": n, "label": lbl} for n in range(30) for lbl in ("type: bug", "topic: config")]
    sample = guided.draw_main(fps, 10)
    assert len(sample) == 10 and all(s["label"] == "type: bug" for s in sample)


def test_warmup_shows_the_answer_and_records_matches(tmp_path):
    # issue 1 is type: question (1 % 5), issue 2 is type: docs
    done, out = session(["2", "1", "y"], plan(), tmp_path)
    assert done == {"warmup": 2, "main": 1}
    assert "Maintainers labelled it: type: question. You matched them." in out
    assert "Different from yours." in out
    assert "matched the maintainers on 1 of 2" in out
    rows = guided.load_answers("me", "warmup", tmp_path)
    assert [r["match"] for r in rows] == [True, False]


def test_session_is_resumable_and_rejects_bad_input(tmp_path):
    done, out = session(["7", "s", "q"], plan(), tmp_path)
    assert "Please type one of" in out
    assert done == {"warmup": 1, "main": 0}
    done, out = session(["4", "n"], plan(), tmp_path)
    assert done == {"warmup": 1, "main": 1}
    assert guided.load_answers("me", "main", tmp_path)[0]["verdict"] == "wrong"


def test_main_question_is_the_plain_language_one(tmp_path):
    _, out = session(["1", "1", "s"], plan(), tmp_path, comments=["Thanks, this is a bug."])
    assert guided.QUESTIONS["type: bug"] in out
    assert "What a pytest maintainer replied" in out
    assert "type: bug" not in out.split("QUESTION 1/1")[1].split("Does the reporter")[0]


def test_long_code_blocks_are_shortened():
    body = "Intro\n```\n" + "\n".join(f"line {k}" for k in range(30)) + "\n```\n<!-- hidden -->End"
    text = guided.shorten_body(body)
    assert "line 7" in text and "line 20" not in text
    assert "22 more lines" in text and "hidden" not in text and text.endswith("End")


def test_only_maintainer_comments_are_shown():
    comments = [
        {"author_association": "NONE", "body": "me too"},
        {"author_association": "MEMBER", "body": "This is   expected behaviour."},
    ]
    assert guided.maintainer_comments(comments) == ["This is expected behaviour."]


def test_score_reports_warmup_precision_and_agreement(tmp_path):
    session(["2", "1", "y"], plan(), tmp_path)  # question: right; docs answered as bug: wrong
    guided._append(
        guided.answers_path("other", "main", tmp_path),
        {"number": 3, "label": "type: bug", "verdict": "correct"},
    )
    result = guided.score("me", plan(), tmp_path)
    assert (result["warmup_matched"], result["warmup_judged"]) == (1, 2)
    assert result["main"]["share_of_fps_correct"] == 1.0
    assert result["main"]["adjusted_precision"] == 1.0  # 10 tp + all 10 fp judged correct
    assert result["agreement"]["other"]["same"] == 1
