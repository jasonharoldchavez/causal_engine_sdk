import sys
sys.path.insert(0,'.')

import numpy as np
import glob
import time

from advanced_math.geometry.violet_engine import VioletEngine
from advanced_math.geometry.takens_embedding import TopologicalStateEngine
from advanced_math.topology.hodge_engine import ASIEHodgeEngine
from advanced_math.analytics.advanced_asie_engine import AdvancedASIEEngine
from causal_engine.models.var_models import PluginLinearVAR
from advanced_math.analytics.state_vector_asie import StateVectorASIE

print("=== ADVANCED MATH ENGINE BATTLE ===")

files=glob.glob("telemanom/data/data/test/*.npy")

results={}

def test_engine(name,fn):
    alerts=0
    channels=0
    start=time.time()

    for f in files:
        try:
            x=np.load(f)
            x=x[:,0] if x.ndim>1 else x

            out=fn(x)

            if out is not None:
                n=len(out) if hasattr(out,"__len__") else 1
                if n:
                    channels+=1
                    alerts+=n

        except Exception:
            pass

    results[name]=(
        channels,
        alerts,
        round(time.time()-start,3)
    )


def violet(x):
    e=VioletEngine()
    return e.build_trajectory_matrix(
        x.reshape(-1,1)
    )


def takens(x):
    e=TopologicalStateEngine()
    return e.construct_phase_space(x)


def state(x):
    e=StateVectorASIE(
        dimensions=1
    )
    for v in x[:1000]:
        e.update(np.array([v]))
    return e.get_state()


def asie(x):
    e=AdvancedASIEEngine(
        num_signals=1
    )
    return e.predict(
        x[-100:].reshape(-1,1)
    )


engines={
"VIOLET":violet,
"TAKENS":takens,
"STATE_VECTOR":state,
"ADVANCED_ASIE":asie
}


for name,fn in engines.items():
    print("Running",name)
    test_engine(name,fn)


print()
for k,v in results.items():
    print(k)
    print("Channels:",v[0])
    print("Outputs:",v[1])
    print("Runtime:",v[2],"sec")
    print()

print("=== COMPLETE ===")
