from tensorly.decomposition import matrix_product_state
import numpy as np

a = np.random.rand(128, 256)
in_mods = [2, 4, 4, 4]
out_mods = [4, 4, 4, 4]
a_ = np.reshape(a, tuple(in_mods[i]*out_mods[i] for i in range(len(in_mods))))
ranks = [1, 2, 2, 2, 1]

l = list()
for i in range(len(in_mods)):
    l.append([out_mods[i] * ranks[i + 1], ranks[i] * in_mods[i]])

res = matrix_product_state(a_, ranks)

for idx_core, shape_core in enumerate(l):
    res[idx_core] = np.reshape(res[idx_core], tuple(shape_core))