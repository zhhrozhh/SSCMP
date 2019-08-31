import pandas as pd
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
KS = scipy.stats.kstest
MMM = pd.read_csv("MMM.csv").Adj_Close
BAC = pd.read_csv("BAC.csv").Adj_Close
St = MMM
dSt = St - St.shift(1)
mut = (St-St.shift(20))/20

# mut = dSt.mean()

st = dSt.rolling(window = 20,center = False).std()
vt = st**2
thetat = vt.rolling(window = 20, center = False).mean()
xit = dvt.rolling(window = 20, center = False).std()

# thetat = vt.mean()
# xit = dvt.std()

kt = 3.06479*xit/thetat
E = (dSt - mut * St)/(St * st)

U = ((E-E.mean())/E.std()).dropna()
P = (dvt - kt*(thetat - vt))/(xit*st)
Q = ((P-P.mean())/P.std()).dropna()


fig = plt.figure(figsize = (12,8))
ax1 = plt.subplot2grid((2,3),(0,0))
ax1.plot(np.arange(-5,5,0.125),scipy.stats.laplace(0,1/np.sqrt(2)).pdf(np.arange(-5,5,0.125)), label = "Laplace(0,sqrt2/2")
ax1.plot(np.arange(-5,5,0.125),scipy.stats.norm.pdf(np.arange(-5,5,0.125)), label = "N(0,1)")
ax1.hist(U,bins = np.arange(-5,5,0.0125),density = True, label = "observed density")
ax1.legend()
ax1.set_title('Z')
ax2 = plt.subplot2grid((2,3),(0,1))
scipy.stats.probplot(U,dist = scipy.stats.laplace(0,1/np.sqrt(2)), plot = ax2)
_, p = KS(U,scipy.stats.laplace(0,1/np.sqrt(2)).cdf)
ax2.set_title("Q-Q plot Laplace(0,sqrt2/2)\nKSp {:1.4E}".format(p))
ax2.xaxis.set_visible(False)


ax3 = plt.subplot2grid((2,3),(0,2))
scipy.stats.probplot(U,dist = "norm", plot = ax3)
_, p = KS(U,scipy.stats.norm().cdf)
ax3.set_title("Q-Q plot N(0,1)\nKSp {:1.4E}".format(p))
ax3.xaxis.set_visible(False)


ax4 = plt.subplot2grid((2,3),(1,0))
ax4.plot(np.arange(-5,5,0.125),scipy.stats.laplace(0,1/np.sqrt(2)).pdf(np.arange(-5,5,0.125)), label = "Laplace(0,sqrt2/2")
ax4.plot(np.arange(-5,5,0.125),scipy.stats.norm.pdf(np.arange(-5,5,0.125)), label = "N(0,1)")
ax4.hist(Q,bins = np.arange(-5,5,0.0125),density = True, label = "observed density")
ax4.set_title('Y')
ax4.legend()
ax5 = plt.subplot2grid((2,3),(1,1))
scipy.stats.probplot(Q,dist = scipy.stats.laplace(0,1/np.sqrt(2)), plot = ax5)
_, p = KS(Q,scipy.stats.laplace(0,1/np.sqrt(2)).cdf)
ax5.set_title("Q-Q plot Laplace(0,sqrt2/2)\nKSp {:1.4E}".format(p))
ax5.xaxis.set_visible(False)


ax6 = plt.subplot2grid((2,3),(1,2))
ax6.xaxis.set_visible(False)
scipy.stats.probplot(Q,dist = "norm", plot = ax6)
_, p = KS(Q,scipy.stats.norm().cdf)
ax6.set_title("Q-Q plot N(0,1)\nKSp {:1.4E}".format(p))

fig.suptitle("MMM")