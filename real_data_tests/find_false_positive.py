import sys
sys.path.insert(0,'.')

import numpy as np
import glob
import csv
import ast

from advanced_math.analytics.advanced_asie_engine import AdvancedASIEEngine

labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]]=ast.literal_eval(r["anomaly_sequences"])

for f in glob.glob("telemanom/data/data/test/*.npy"):

    chan=f.split("/")[-1].replace(".npy","")

    x=np.load(f)
    x=x[:,0] if x.ndim>1 else x

    e=AdvancedASIEEngine(num_signals=1)

    try:
        out=e.predict(x[-100:].reshape(-1,1))

        detected=len(out)

        truth=len(labels.get(chan,[]))

        if detected and not truth:
            print("FALSE POSITIVE:")
            print(chan)
            print("Output:")
            print(out)

    except Exception:
        pass
