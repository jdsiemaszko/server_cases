import numpy as np

class Dipole():
    def __init__(self, Gamma, nu, x1, y1, x2, y2, R, uxInf, uyInf):
        self.Gamma = Gamma
        self.nu = nu
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.R = R
        self.uxInf = uxInf
        self.uyInf = uyInf

    def vorticity(self, xEval, yEval):
        r1x = xEval - self.x1
        r1y = yEval - self.y1
        rsq1 = (r1x * r1x + r1y * r1y) / self.R / self.R
        del r1x, r1y

        r2x = xEval - self.x2
        r2y = yEval - self.y2
        rsq2 = (r2x * r2x + r2y * r2y) / self.R / self.R
        del r2x, r2y

        return self.Gamma * (1 - rsq1) * np.exp(-rsq1)\
            - self.Gamma * (1 - rsq2) * np.exp(-rsq2)
    

