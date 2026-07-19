import sys
sys.path.insert(0,'.')

import numpy as np

from advanced_math.analytics.advanced_asie_engine import AdvancedASIEEngine

x=np.load("telemanom/data/data/test/A-1.npy")

if x.ndim>1:
    x=x[:,0]

e=AdvancedASIEEngine(num_signals=1)

for i in [100,200,500,1000,2000,4000]:
    if i < len(x):
        out=e.predict(
            x[i-100:i].reshape(-1,1)
        )

        print("WINDOW END:",i)
        print("TYPE:",type(out))
        print("VALUE:",out)
        print("LEN:",len(out) if hasattr(out,"__len__") else "scalar")
        print()
