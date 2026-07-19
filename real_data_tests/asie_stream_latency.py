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
        labels[r["chan_id"]]=ast.literal_eval(r["chan_id"] and r["anomaly_sequences"])

for f in glob.glob("telemanom/data/data/test/*.npy"):

    chan=f.split("/")[-1].replace(".npy","")

    if labels.get(chan):

        x=np.load(f)
        x=x[:,0] if x.ndim>1 else x

        e=AdvancedASIEEngine(num_signals=1)

        detected=None

        window=[]

        for i,v in enumerate(x):
            window.append(v)

            if len(window)>=100:
                out=e.predict(
                    np.array(window[-100:]).reshape(-1,1)
                )

                if len(out):
                    detected=i
                    break

        if detected is not None:
            print(chan, detected)

