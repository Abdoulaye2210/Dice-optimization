import numpy as np
import math
import matplotlib.pyplot as plt

def Douglas(num_times):
    S0 = 25.
    r = 0.1
    sigma = 0.20
    T = 1
    dt = T / num_times

    z = np.random.standard_normal((num_times+1,1))
    z -= z.mean()
    z /= z.std()
    S = np.zeros((num_times+1,1))
    S[0] = S0
    for i in range(1, num_times+1):
      S[i] = S[i-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * z[i]) ### Goemetric Brownian motion
    #plt.plot(S[:, :1]);
    return S[1:num_times+1]
