import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict_cosmos_all_models.py"
SPEC = importlib.util.spec_from_file_location("predict_cosmos_all_models", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_experimental_values_and_nearest():
    assert MODULE.parse_experimental_values("-4.511 | -4.352") == [-4.511, -4.352]
    assert MODULE.nearest_experimental("-4.511 | -4.352", -4.40) == -4.352


def test_nearest_tie_is_deterministic():
    assert MODULE.nearest_experimental("-5 | -3", -4.0) == -5.0
