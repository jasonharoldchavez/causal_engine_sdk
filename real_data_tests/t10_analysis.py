import numpy as np

x=np.load("telemanom/data/data/test/T-10.npy")

if x.ndim>1:
    x=x[:,0]

window=x[-100:]

slope=np.polyfit(
    np.arange(len(window)),
    window,
    1
)[0]

print("Last window slope:",slope)

if slope < -0.005:
    print("DOWNWARD DRIFT DETECTED")
