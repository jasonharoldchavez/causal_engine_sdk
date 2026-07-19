import sys
sys.path.insert(0,'.')
sys.path.insert(0,'../oses_test/streaming_guardian')

import numpy as np
import glob

from advanced_math.analytics.advanced_asie_engine import AdvancedASIEEngine
from detectors.models import CUSUMDecisionEngine, EWMADetector

for f in glob.glob("telemanom/data/data/test/*.npy"):

    chan=f.split("/")[-1].replace(".npy","")

    x=np.load(f)
    x=x[:,0] if x.ndim>1 else x

    results=[]

    a=AdvancedASIEEngine(num_signals=1)
    results.append(len(a.predict(x[-100:].reshape(-1,1)))>0)

    c=CUSUMDecisionEngine()
    results.append(any(c.update(v) for v in x))

    e=EWMADetector()
    results.append(any(e.update(v) for v in x))

    if len(set(results)) > 1:
        print(chan, results)
