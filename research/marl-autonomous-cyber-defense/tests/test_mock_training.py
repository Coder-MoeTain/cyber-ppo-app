import numpy as np

from autocd.algorithms.trainer import TrainConfig, train_mock


def test_mock_training_smoke():
    model, returns = train_mock(TrainConfig(episodes=2, seed=13, update_epochs=1))
    assert len(returns) == 2
    assert np.isfinite(returns).all()
    assert len(model.actors) == 3
