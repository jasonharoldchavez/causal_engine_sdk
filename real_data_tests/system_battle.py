import glob
import numpy as np
import time

print("=== TELEMETRY SYSTEM BATTLE ===")

files=glob.glob("telemanom/data/data/test/*.npy")

systems={}

def std_detector(x):
    m=np.mean(x[:500])
    s=np.std(x[:500])+1e-9
    return np.where(abs(x-m)>5*s)[0]

def cusum_detector(x):
    out=[]
    c=0
    m=0
    for i,v in enumerate(x):
        c=max(0,c+(v-m-0.5))
        m=.999*m+.001*v
        if c>10:
            out.append(i)
            c=0
    return np.array(out)

def quantum_detector(x):
    amp=np.abs(x)
    phase=np.angle(np.exp(1j*x))
    coherence=np.abs(np.mean(np.exp(1j*phase)))
    score=np.abs(amp-np.mean(amp))*(1+coherence)
    return np.where(score>np.mean(score)+5*np.std(score))[0]

systems["STD"]=std_detector
systems["CUSUM"]=cusum_detector
systems["QUANTUM"]=quantum_detector

for name,fn in systems.items():
    total_alerts=0
    channels=0
    start=time.time()

    for f in files:
        x=np.load(f)
        x=x[:,0] if x.ndim>1 else x
        alerts=fn(x)

        if len(alerts):
            channels+=1
        total_alerts+=len(alerts)

    print()
    print(name)
    print("Channels detected:",channels,"/82")
    print("Total alerts:",total_alerts)
    print("Runtime:",round(time.time()-start,3),"sec")

print()
print("=== COMPLETE ===")
