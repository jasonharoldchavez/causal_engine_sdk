import sys
sys.path.insert(0,'.')

import numpy as np

from advanced_math.analytics.advanced_asie_engine import AdvancedASIEEngine

x=np.load("telemanom/data/data/test/A-1.npy")

if x.ndim>1:
    x=x[:,0]

e=AdvancedASIEEngine(num_signals=1)

for i in range(100,len(x),500):
    score=e.predict(
        x[i-100:i].reshape(-1,1)
    )[0]

    print("END:",i,"SCORE:",score)
