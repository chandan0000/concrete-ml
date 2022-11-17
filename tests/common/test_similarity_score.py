"""Tests for the r2 score test """
import numpy
import pytest

BIG_OFFSET = 10000
BIG_STDEV = 100
SMALL_VALUE = 0.00001
SMALL_STDEV = SMALL_VALUE
SMALL_OFFSET = SMALL_VALUE


@pytest.mark.parametrize("gt_range", [SMALL_STDEV, BIG_STDEV])
@pytest.mark.parametrize("gt_stdev", [SMALL_STDEV, BIG_STDEV])
@pytest.mark.parametrize(
    "predicted_params",
    zip(
        [None, SMALL_STDEV, SMALL_STDEV, BIG_STDEV, BIG_STDEV],
        [0, SMALL_OFFSET, BIG_OFFSET, SMALL_OFFSET, BIG_OFFSET],
    ),
)
@pytest.mark.parametrize("num_values", [1000])
def test_r2(
    gt_range,
    gt_stdev,
    predicted_params,
    num_values,
    check_r2_score,
):
    """Test our modified r2 test on various data distributions"""

    # Generate a uniform distribution and add a normal residual on top
    gt_values = numpy.random.uniform(size=(num_values,)) * gt_range
    gt_normal_residuals = numpy.random.normal(0, gt_stdev, size=(num_values,))
    gt_values += gt_normal_residuals

    predicted_stdev, predicted_offset = predicted_params

    if predicted_stdev is not None:
        # Correlated predicted value
        pred_normal_residuals = numpy.random.normal(0, predicted_stdev, size=(num_values,))
        predicted = gt_values + pred_normal_residuals
    else:
        # No correlation between gt and predicted
        predicted = numpy.random.uniform(size=(num_values,))

    predicted += predicted_offset

    should_fail = (
        predicted_stdev is None
        or predicted_stdev > SMALL_VALUE
        or predicted_offset > SMALL_VALUE
    )

    if should_fail:
        with pytest.raises(AssertionError):
            check_r2_score(gt_values, predicted)
    else:
        check_r2_score(gt_values, predicted)
