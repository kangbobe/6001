import numpy as np
import pytest
from src.metrics import binary_metrics


def test_confusion_and_ranking_metrics():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9])
    assert (m['tn'], m['fp'], m['fn'], m['tp']) == (1, 1, 1, 1)
    assert m['f1'] == m['accuracy'] == 0.5
    assert m['roc_auc'] == 0.75
    assert m['average_precision'] == pytest.approx((1 + 2/3)/2)


def test_threshold_does_not_change_ranking_metrics():
    a = binary_metrics([0, 0, 1, 1], [.1, .8, .4, .9], .5)
    b = binary_metrics([0, 0, 1, 1], [.1, .8, .4, .9], .3)
    assert a['average_precision'] == b['average_precision']
    assert b['recall'] == 1


@pytest.mark.parametrize('p', [[.1, np.nan], [-.1, .9], [.1, 1.1]])
def test_bad_probabilities_fail(p):
    with pytest.raises(ValueError):
        binary_metrics([0, 1], p)
