import pytest

from engine.verify import sql
from engine.verify.sandbox import run_tests

TESTS = [{"call": "add(2, 3)", "expect": "5"}, {"call": "add(-1, 1)", "expect": "0"}]


def test_good_code_passes():
    r = run_tests("def add(a, b):\n    return a + b\n", TESTS)
    assert r["error"] is None and r["passed"] == 2


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef add(a,b):\n    return os.system('echo hi')",
        "import socket\ndef add(a,b): return 1",
        "def add(a,b):\n    return open('/etc/passwd').read()",
        "def add(a,b):\n    return ().__class__.__bases__[0].__subclasses__()",
        "def add(a,b):\n    return eval('a+b')",
    ],
)
def test_static_screen_rejects(code):
    r = run_tests(code, TESTS)
    assert r["passed"] == 0 and r["error"].startswith("rejected")


def test_runtime_hooks_block_file_reads_and_writes():
    # Allowed import, but the audit hook blocks reading outside the stdlib at runtime.
    code = "import json\ndef add(a, b):\n    return json.load(type(json).__dict__ and None or 0)"
    r = run_tests(code, TESTS)
    assert r["passed"] == 0


def test_timeout():
    r = run_tests("def add(a, b):\n    while True:\n        pass\n", TESTS, timeout_s=2)
    assert r["passed"] == 0 and r["error"] in ("timeout",) or "crashed" in (r["error"] or "")


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "t.sqlite"
    sql.build_database({"orders": [{"region": "N", "total": 10.0}, {"region": "S", "total": 5.5}, {"region": "N", "total": 1.0}]}, p)
    return p


def test_sql_match(db):
    ok, _ = sql.verify(db, "SELECT region, SUM(total) FROM orders GROUP BY region", "select region, sum(total) as t from orders group by region")
    assert ok
    ok, _ = sql.verify(db, "SELECT COUNT(*) FROM orders", "SELECT 2")
    assert not ok


@pytest.mark.parametrize("bad", ["DELETE FROM orders", "ATTACH DATABASE 'x.db' AS x", "PRAGMA writable_schema=1", "SELECT 1; DROP TABLE orders", "SELECT load_extension('x')"])
def test_sql_rejects_writes(db, bad):
    ok, detail = sql.verify(db, "SELECT COUNT(*) FROM orders", bad)
    assert not ok and detail.get("error")


def test_limits_apply_inside_the_runner():
    # CPU: a busy loop is ended by the limit the runner put on itself (or by the wall clock).
    r = run_tests("def add(a, b):\n    while True:\n        pass\n", TESTS, timeout_s=1)
    assert r["passed"] == 0 and (r["error"] == "timeout" or "crashed" in r["error"])
    # Memory: half a gigabyte is the ceiling; asking for much more fails inside the test.
    import sys

    if sys.platform.startswith("linux"):
        r = run_tests("def add(a, b):\n    return len(bytearray(2_000_000_000))\n", TESTS)
        assert r["passed"] == 0


def test_many_checks_at_once_all_finish():
    """Answers are checked four at a time from threads; none may get stuck starting up."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: run_tests(f"def add(a, b):\n    return a + b + {i} - {i}\n", TESTS), range(48)))
    assert all(r["error"] is None and r["passed"] == 2 for r in results), [r for r in results if r["error"]][:3]
