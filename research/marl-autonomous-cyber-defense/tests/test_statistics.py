import pytest
from autocd.statistics import paired_bootstrap_ci,paired_permutation_test,holm_bonferroni

def test_bootstrap_ci():
    r=paired_bootstrap_ci([3,4,5],[1,2,3],n_boot=1000,seed=7)
    assert r["mean_difference"]==pytest.approx(2.) and r["ci_low"]<=2.<=r["ci_high"]
def test_permutation():
    r=paired_permutation_test([5,6,7,8],[1,2,3,4],n_permutations=2000)
    assert r["mean_difference"]==pytest.approx(4.) and 0<r["p_value"]<=1
def test_holm_order():assert holm_bonferroni([.01,.04,.03])==pytest.approx([.03,.06,.06])
def test_bad_pairs():
    with pytest.raises(ValueError):paired_bootstrap_ci([1],[1])
