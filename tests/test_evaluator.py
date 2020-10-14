import pandas as pd
from pandas._testing import assert_frame_equal

from bluepyemodel.ais_synthesis.tools import evaluate_combos


def _evaluation_function(combo):
    """Mock evaluation function."""
    return {"result_orig": combo["value"], "result_10": 10.0 * combo["value"]}


def test_evaluate_combos():
    """Test combos evaluator on a trivial example."""
    combos = pd.DataFrame()
    combos.loc[0, "name"] = "test1"
    combos.loc[0, "value"] = 1.0
    combos.loc[1, "name"] = "test2"
    combos.loc[1, "value"] = 2.0

    expected_result_combos = combos.copy()
    expected_result_combos["exception"] = ""
    expected_result_combos["to_run_result_orig"] = 0
    expected_result_combos["to_run_result_10"] = 0
    expected_result_combos.loc[0, "value"] = 1.0
    expected_result_combos.loc[1, "value"] = 2.0
    expected_result_combos.loc[0, "result_orig"] = 1.0
    expected_result_combos.loc[1, "result_orig"] = 2.0
    expected_result_combos.loc[0, "result_10"] = 10.0
    expected_result_combos.loc[1, "result_10"] = 20.0

    new_columns = [["result_orig", 0.0], ["result_10", 0.0]]
    result_combos = evaluate_combos(combos, _evaluation_function, new_columns)
    assert_frame_equal(result_combos, expected_result_combos, check_like=True)
