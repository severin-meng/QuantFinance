import numpy as np
import matplotlib.pyplot as plt

start = 0.5
end = 2.0
steps = 30

logspace = np.exp(np.linspace(np.log(start), np.log(end), steps))

linspace = np.linspace(start, end, steps)
ones = np.ones_like(logspace)

plt.figure()
plt.scatter(logspace, ones, label="log")
plt.scatter(linspace, 0.9*ones, label="lin")
plt.legend()
plt.show()