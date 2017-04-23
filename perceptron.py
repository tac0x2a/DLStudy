import numpy as np
from numpy.random import *

# Injection?
def forward(x = [], W = [], th = 0):
    d = np.dot(np.array(x), np.array(W))
    return 1 if(d > th) else -1

def error(x, W, t):
    return max(0, -1 * t * np.dot(x, W))

def append_bias(X, W):
    x = np.array([ np.append(x, 1) for x in X ])
    w = np.append(W, rand())
    return[x, w]

def next_w(x, W, t, lo = 0.1):
    return W  + (lo * t * x)

# ---------------------------------------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([-1, -1, -1, 1]) # AND gate. -1:false, 1:true
W = np.array([ rand()-0.5 for x in range(0, len(X[0])) ])

# AppendBias
Xb, W = append_bias(X, W)
W0 = W
print(Xb,Y,W)

errors = [] # for Visualize

for i in range(0, 100):
    error_total = sum([ error(Xb[idx], W, Y[idx]) for idx in range(0, len(Xb))])
    errors.append(error_total) # for Visualize
    if error_total <= 0:
        print("Finish at iteration " + str(i))
        break

    for idx in range(0, len(Xb)):
        x = Xb[idx]
        f = forward(x,W)
        t = Y[idx]
        e = error(x,W,t)
        if e > 0 :
            W = next_w(x, W, t, 0.1)
        print( str(Xb[idx]) + "." + str(W) + "=" + str(f) + ":" + str(t) + ":" + str(f) + ":" + str(e)  )

# ---------------------------------------------------------------
# Visualize
print(errors)
import matplotlib.pyplot as plt

plt.xlim(-1,2.5)
plt.ylim(-1,2.5)

plt.subplot(211)
plt.plot(errors)

plt.subplot(212)
xx, yy = np.transpose(X)
plt.scatter(xx, yy)

x = np.linspace(-0.5, 1.5)
y0 = -(W0[0]*x + W0[2]) / W0[1]
plt.plot(x,y0)

# W[0]*x + W[1]*y + W[2]= 0
# W[1]*y = -W[0]*x - W[2]
# y = -(W[0]*x + W[2]) / W[1]
y = -(W[0]*x + W[2]) / W[1]

plt.plot(x,y)
plt.show()
