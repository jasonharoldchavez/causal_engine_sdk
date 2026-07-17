import numpy as np


class CausalInvarianceValidator:

    def compare(self,g1,g2):

        a=g1.flatten()
        b=g2.flatten()

        denom=np.linalg.norm(a)*np.linalg.norm(b)

        if denom==0:
            return 0

        return np.dot(a,b)/denom
