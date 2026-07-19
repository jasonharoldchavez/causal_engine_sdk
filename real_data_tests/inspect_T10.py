import numpy as np
import matplotlib.pyplot as plt

x=np.load("telemanom/data/data/test/T-10.npy")

if x.ndim>1:
    x=x[:,0]

print("Length:",len(x))
print("Min:",np.min(x))
print("Max:",np.max(x))
print("Mean:",np.mean(x))
print("Std:",np.std(x))

print("\nLast 20 samples:")
print(x[-20:])

plt.plot(x)
plt.title("T-10 Signal")
plt.savefig("T10_signal.png")

