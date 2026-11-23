# Event verification regression tests

Run from the vendored `vlmevalkit` directory with Python 3.12 and the
CPU evaluator dependencies (`pytest`, `pandas`, `scikit-learn`, `numpy`,
and imports required by `vlmeval`). No model weights or video assets are
needed.

```sh
python -m pytest -c /dev/null --confcutdir=tests -q tests/test_event_verification.py
```

The `-c /dev/null` and `--confcutdir=tests` options intentionally isolate
these unit tests from Cosmos Framework's root-level pytest configuration and
integration-test fixtures. They do not constitute a full framework or
model-serving test run. The tests exercise the real evaluator and CSV I/O.
