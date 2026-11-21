import numpy as np
from scipy.ndimage import distance_transform_edt

from floodunc import ml, validate as val
from floodunc.factors import FACTOR_NAMES, generate_catchment


def test_block_cv_no_leakage_and_full_coverage():
    shape, bs, buf = (60, 60), 15, 4
    blocks = ml.block_ids(shape, bs).reshape(-1)
    seen = np.zeros(blocks.size, int)
    for tr, te in ml.spatial_block_splits(shape, bs, 4, buf, seed=1):
        assert np.intersect1d(tr, te).size == 0
        # no block is shared between train and test
        assert np.intersect1d(np.unique(blocks[tr]), np.unique(blocks[te])).size == 0
        # every training cell is farther than the buffer from every test cell
        test_mask = np.zeros(blocks.size, bool)
        test_mask[te] = True
        d = distance_transform_edt(~test_mask.reshape(shape)).reshape(-1)
        assert d[tr].min() > buf
        seen[te] += 1
    assert (seen == 1).all()  # each cell tested exactly once


def test_random_split_does_share_blocks_which_is_the_leak():
    blocks = ml.block_ids((60, 60), 15).reshape(-1)
    tr, te = next(ml.random_splits(blocks.size, 4, 0))
    assert np.intersect1d(np.unique(blocks[tr]), np.unique(blocks[te])).size > 0


def test_catchment_generator():
    a = generate_catchment(n=60, seed=7)
    b = generate_catchment(n=60, seed=7)
    c = generate_catchment(n=60, seed=8)
    assert np.array_equal(a.inventory, b.inventory)
    assert not np.array_equal(a.raw["elevation"], c.raw["elevation"])
    st = a.criteria_stack()
    assert st.shape == (len(FACTOR_NAMES), 60, 60)
    assert st.min() >= 0 and st.max() <= 1
    assert abs(a.inventory.mean() - 0.10) < 0.03
    assert 0 < a.true_susceptibility.min() and a.true_susceptibility.max() < 1


def test_oof_predictions_beat_chance_on_synthetic_data():
    cat = generate_catchment(n=60, seed=2)
    X, y = cat.design_matrix(), cat.labels()
    splits = list(ml.spatial_block_splits(cat.shape, 15, 4, 3, 0))
    oof = ml.cross_val_predict_splits(ml.make_models(0)["lr"], X, y, splits)
    assert val.auc(y, oof) > 0.6


def test_validation_metrics():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    assert val.auc(y, y + 0.0) == 1.0
    assert abs(val.auc(y, rng.random(5000)) - 0.5) < 0.05
    x, h, a = val.success_rate_curve(y, y + rng.random(5000) * 0.01)
    assert x[0] == 0 and h[-1] == 1 and a > 0.9
    p, r = val.precision_recall_top(y, y + rng.random(5000) * 0.01, 0.2)
    assert r == 1.0 and abs(p - y.sum() / (0.2 * y.size)) < 0.01
