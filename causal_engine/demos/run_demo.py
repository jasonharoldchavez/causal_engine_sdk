import numpy as np

from causal_engine.simulators.sem import StructuralEquationSimulator


sim=StructuralEquationSimulator(4)

data=[]

for t in range(100):
    noise=np.random.randn(4)*0.05
    z=np.sin(t*.05)

    data.append(sim.step(noise,z))


data=np.array(data)

print("CAUSAL ENGINE DEMO COMPLETE")
print("Generated stream shape:",data.shape)
print(data[-1])
