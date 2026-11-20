import json

from lorm.eval.task_bpb import gold_rows, item_bpb


def test_gold_rows_keep_the_gold_choice_of_conditional_requests(tmp_path):
    rows = [dict(doc_id=0, idx=0, label=1, context="Q", continuation=" a"),
            dict(doc_id=0, idx=1, label=1, context="Q", continuation=" b"),
            dict(doc_id=1_000_000, idx=1, label=1, context="Answer:", continuation=" b"),
            dict(doc_id=3, idx=None, label="x = 1", context="Solve", continuation=" x = 1")]
    path = tmp_path / "task.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [r["continuation"] for r in gold_rows(path)] == [" b", " x = 1"]


def test_item_bpb_divides_by_utf8_bytes():
    assert abs(item_bpb(-2.0, "é!") - 2 / 0.6931471805599453 / 3) < 1e-12
