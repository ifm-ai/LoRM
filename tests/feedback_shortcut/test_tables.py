from works.feedback_shortcut.tables import TABLES


def test_every_table_builds_from_the_shipped_results():
    for label, build in TABLES.items():
        rows = build()
        assert any("&" in r for r in rows), label
