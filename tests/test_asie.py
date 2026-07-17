import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from causal_engine.asie import ASIEEngine


def test_asie_prediction():
    engine = ASIEEngine(3)

    data = np.random.rand(50,3)

    prediction = engine.predict(data)

    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()


if __name__ == "__main__":
    test_asie_prediction()
    print("ASIE TEST PASSED")
