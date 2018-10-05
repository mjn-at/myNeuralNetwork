"""
creates a colored output of an array
"""

import numpy as np
import matplotlib.pyplot as plt

a = np.random.rand(3,3) - 0.5
print(a)

plt.imshow(a, interpolation="nearest")

plt.show()
