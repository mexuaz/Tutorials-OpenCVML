import time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.metrics import mean_squared_error


def f(X): return np.sin(X) + np.cos(3*X)


np.random.seed(seed=int(time.time()))  # seeding random generator

a = -1*np.pi  # domain min
b = np.pi  # domain max


x = (b - a) * np.random.random_sample(150) + a  # input
y = f(x) + np.random.uniform(-.4, .4, (150,))  # output


SplitRatio = .7 # split input samples of data
border = int(len(x)*SplitRatio)
xtr, ytr = x[:border,], y[:border,]  # Training samples
xte, yte = x[border:,], y[border:,]  # Testing samples


fig = plt.plot(x, y, 'b.')  # plotting training samples


# Constructing a Multilayer Perceptron
model = Sequential()    # Feed forward Network
model.add(Dense(4, input_dim=1, activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(1))  # output layer

# stochastic gradient descent (sgd) optimization
model.compile(loss='mean_squared_error', optimizer='sgd')
# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

tm = time.time()
hist = model.fit(xtr, ytr, epochs=10000, verbose=0)
tm_train = time.time() - tm

xr = np.arange(a, b, .001)

tm = time.time()
results = model.predict(xr)  # predicting values over the function domain
tm_predict = time.time() - tm

print("RMS: ", mean_squared_error(xte, yte))  # calculating RMSE on test samples
# print(model.evaluate(xtr, ytr)) # print loss and accuracy
print("Training Time: %.4f s" % tm_train)
print("Prediction Time: %.4f ms" % (tm_predict * 1000))


fig = plt.plot(xr, f(xr), 'r')  # plotting the real function
plt.plot(xr, results, 'g')  # plotting the predicted values over the function domain
plt.show()

'''
RMS:  1.99770915462
Training Time: 78.9976 s
Prediction Time: 210.7198 ms
'''
