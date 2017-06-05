# ----------------------------------------------------
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # Error (Output CrossEntropy)
        self.y = None # Out ot Softmax
        self.t = None # Teaching data

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x) # Todo
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dz):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

# ----------------------------------------------------
class AffineLayer:
    def __init__(self):
        self.x = None
        self.W = None
        self.b = None

    # z = x*W + b
    def forward(self, x, W, b):
        self.x = x
        self.W = W
        self.b = b
        z = np.dot(x,w) + b
        return z

    # dx = dz * W^T
    def backward(self, dz):
        dx = np.dot(dz, self.W.T)
        return dx

# ----------------------------------------------------
class ReLULayer:
    def __init__(self):
        self.mask = None

    # z = 0 if x <= 0
    # z = x if x > 0
    def forward(self, x): # x expected numpy.array
        self.mask = (x <= 0) # numpy.array
        z = x.copy()
        z[self.mask] = 0
        return z

    # dx = 0  if x <= 0
    # dx = dz if x > 0
    def backward(self, dz): # z expected numpy.array
        dz[self.mask] = 0
        dx = dz
        return dx

# ----------------------------------------------------
class SigmoidLayer:
    def __init__(self):
        self.z = None

    # z = 1 / (1 + exp(-x))
    def forward(self, x):
        self.z = 1.0 / (1.0 + exp(-x))
        return self.z

    # dx = dz * z*(1-z)
    def backward(self, dz):
        dx = dz * self.out * (1.0 - self.out)
        return dx

# ----------------------------------------------------
# Input(x * y) => Output(z)
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        z = x * y
        return z

    def backward(self, dz):
        dx = dz * y
        dy = dz * x
        return dx, dy

# ----------------------------------------------------
# Input(x + y) => Output(z)
class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y

    def backward(self, dz):
        dx = dz * 1
        dy = dz * 1
        return dx, dy
