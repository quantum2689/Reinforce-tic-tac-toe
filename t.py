import numpy as np

bord = np.zeros((3, 3))
bord[1, 1] = 1  # example occupied cell

logits = np.random.randint(0, 10, size=(3, 3)).astype(float)
print("Original logits:")
print(logits)
print('+===========+')

# Mask occupied cells with -inf
logits[bord != 0] = -np.inf
print("Masked logits (flat):")
print(logits.reshape(1, 9).tolist()[0])