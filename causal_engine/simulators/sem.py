import numpy as np


class StructuralEquationSimulator:

    def __init__(self,num_signals=4):
        self.num_signals=num_signals
        self.history=[]


    def step(self,noise,z):
        if len(self.history)<2:
            state=noise
        else:
            last=self.history[-1]

            state=np.zeros(self.num_signals)

            state[0]=np.tanh(.7*last[0])+noise[0]

            state[1]=(
                .4*last[1]
                +.6*(last[0]**2)
                +noise[1]
            )

            state[2]=.5*last[2]+.8*z+noise[2]

            state[3]=.3*last[3]+.8*z+noise[3]

        self.history.append(state)
        return state
