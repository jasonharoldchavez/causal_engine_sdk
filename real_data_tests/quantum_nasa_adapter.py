import numpy as np
import glob
import time

print("=== QUANTUM NASA TELEMETRY ADAPTER TEST ===")

files = glob.glob("telemanom/data/data/test/*.npy")

TP_STREAMS = 0
TOTAL = 0

start=time.time()

for f in files:
    data=np.load(f)

    x=data[:,0] if data.ndim>1 else data

    # quantum-inspired feature extraction
    amplitude=np.abs(x)
    phase=np.angle(np.exp(1j*x))

    # coherence score
    coherence=np.abs(
        np.mean(
            np.exp(1j*phase)
        )
    )

    score=np.abs(
        amplitude-np.mean(amplitude)
    )*(1+coherence)

    alerts=np.where(
        score > np.mean(score)+5*np.std(score)
    )[0]

    TOTAL+=len(x)

    if len(alerts):
        TP_STREAMS+=1

elapsed=time.time()-start

print("Channels tested:",len(files))
print("Samples:",TOTAL)
print("Channels with detections:",TP_STREAMS)
print("Runtime:",round(elapsed,3),"seconds")
print("=== COMPLETE ===")
