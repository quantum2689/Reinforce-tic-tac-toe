import numpy as np

rng = np.random.default_rng()

class network:
    def __init__(self):
        self.w1 = rng.random((9,256))*0.02
        self.b1 = rng.random((1,256))*0.02
        self.w2 = rng.random((256,9))*0.02
        self.b2 = rng.random((1,9))*0.02
    def relu(self,x):
        return np.maximum(0,x)
    def forward(self,x):
        x = x.reshape(1,9)
        #(1,9) @ (9,256) = (1,256)
        x = x @ self.w1
        x = x + self.b1
        # relu function
        x = self.relu(x)            
        # (1,256) @ (256,9) = (1,9)
        x = x @ self.w2
        x = x + self.b2

        return x.reshape(3,3)
        
#net = network()
#x = rng.random((3,3))
#print(net.forward(x))