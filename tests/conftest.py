import pandas as pd
import pytest

from soccer_quant import synthetic
from soccer_quant.config import LEAGUES
from soccer_quant.dataset import Dataset
from soccer_quant.storage import Store

TODAY = pd.Timestamp("2026-09-28 12:00", tz="UTC")


@pytest.fixture(scope="session")
def synthetic_store():
    store = Store(":memory:")
    truth = {key: synthetic.build(store, LEAGUES[key], today=TODAY, seasons=2) for key in ("epl", "ligamx")}
    store.truth = truth
    return store


@pytest.fixture(scope="session")
def epl_data(synthetic_store):
    return Dataset.from_store(synthetic_store, LEAGUES["epl"])


@pytest.fixture(scope="session")
def ligamx_data(synthetic_store):
    return Dataset.from_store(synthetic_store, LEAGUES["ligamx"])
