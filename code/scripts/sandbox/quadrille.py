from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD

lst_coor = [[0, 0], [0, 1], [1, 1], [1, 0]]

lst_data = []
lst_labels = []
lst_labels_1h = []
size_each = 100
dim = 2
for i, coor in enumerate(lst_coor):
    if i%2 == 0:
        y_1h = np.array([0, 1])
        y = 1

    else:
        y = 0
        y_1h = np.array([1, 0])
    x = np.random.randn(size_each, dim) + np.array(coor) *10
    lst_data.append(x)
    lst_labels.append([y] * size_each)
    lst_labels_1h.append([y_1h] * size_each)

X = np.vstack(lst_data)
Y = np.array(lst_labels).flatten()
Y_1h = np.vstack(lst_labels_1h)

X = X - np.mean(X, axis=0)
X = X / np.std(X, axis=0)

shuffle = np.random.permutation(len(X))
X = X[shuffle]
Y = Y[shuffle]

plt.scatter(X[Y==0][:, 0], X[Y==0][:, 1], color="grey")
plt.scatter(X[Y==1][:, 0], X[Y==1][:, 1], color="grey")
# plt.show()

nb_lines = 2
model = Sequential()
model.add(Dense(nb_lines, activation='relu', input_shape=(2,)))
model.add(Dense(1, activation='tanh'))
sgd = SGD(lr=0.1)
model.compile(sgd, loss='binary_crossentropy', metrics=['accuracy'])

model.layers[0].set_weights([np.array([
    [-2, 0.1],
    [-1, 0.1]
]), np.array([0, 0])])

model.fit(x=X, y=Y, batch_size=10, epochs=10)
print(model.evaluate(x=X, y=Y))

weights = model.layers[0].get_weights()
print(weights[0].shape)
xx = np.linspace(min(X[:, 0]), max(X[:, 0]), 50)

color = [
    "red",
    "blue"
]

absolute_max = -1
lst_yy = []
for idx_w in range(weights[0].shape[1]):
    a, b = weights[0][:, idx_w]
    c = weights[1][idx_w]
    print(a, b, c)
    func = lambda x: a/-b * x + c
    yy = [func(x) for x in xx]
    plt.plot(xx, yy, color="black")
    max_yy = max(yy)
    max_input = max(X[:, 1])
    absolute_max = max(absolute_max, max_yy, max_input)
    lst_yy.append(yy)
    plt.text(xx[0] , yy[0]+ 0.1, f"$w_{idx_w}$")

for idx_w in range(weights[0].shape[1]):
    plt.fill_between(xx, lst_yy[idx_w], absolute_max , facecolor=color[idx_w], alpha=0.5)


plt.show()




