import os

import pytest

from experiment.config import load_config


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_SLOW=1 to run tests that load a tiny model")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def smoke_cfg(tmp_path):
    """The smoke overlay with every root pointed at a temp dir."""
    return load_config(
        ["configs/smoke.yaml"],
        overrides=[
            f"experiment.data_root={tmp_path / 'data'}",
            f"experiment.results_root={tmp_path / 'results'}",
            f"experiment.checkpoint_root={tmp_path / 'checkpoints'}",
            f"experiment.log_root={tmp_path / 'logs'}",
            "experiment.tensorboard=false",
        ],
    )
