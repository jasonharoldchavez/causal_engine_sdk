import sys
sys.path.insert(0,'.')

import numpy as np
import glob
import csv
import ast
import time

from advanced_math.geometry.takens_embedding import TopologicalStateEngine
from advanced_math.analytics.advanced_asie_engine import AdvancedASIEEngine
from advanced_math.analytics.state_vector_asie import StateVectorASIE

labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]]=ast.literal_eval(r["anomaly_sequences"])

files=glob.glob("telemanom/data/data/test/*.npy")


def run(name, detector):

    TP=FP=FN=0
    start=time.time()

    for f in files:

        chan=f.split("/")[-1].replace(".npy","")

        x=np.load(f)
        x=x[:,0] if x.ndim>1 else x

        try:
            scores=detector(x)

            if np.isscalar(scores):
                scores=[scores]

            detected=len(scores)

            truth=len(labels.get(chan,[]))

            if detected and truth:
                TP+=1
            elif detected:
                FP+=1

            if truth and not detected:
                FN+=truth

        except:
            pass

    p=TP/(TP+FP) if TP+FP else 0
    r=TP/(TP+FN) if TP+FN else 0
    f1=2*p*r/(p+r) if p+r else 0

    print()
    print(name)
    print("TP",TP)
    print("FP",FP)
    print("FN",FN)
    print("Precision",round(p,4))
    print("Recall",round(r,4))
    print("F1",round(f1,4))
    print("Runtime",round(time.time()-start,3))


def takens_detector(x):
    e=TopologicalStateEngine()
    phase=e.construct_phase_space(x)
    return phase


def state_detector(x):
    e=StateVectorASIE(dimensions=1)
    for v in x[:1000]:
        e.update(np.array([v]))
    return e.get_state()


def asie_detector(x):
    e=AdvancedASIEEngine(num_signals=1)
    return e.predict(x[-100:].reshape(-1,1))


run("TAKENS",takens_detector)
run("STATE_VECTOR",state_detector)
run("ADVANCED_ASIE",asie_detector)

print("=== COMPLETE ===")
