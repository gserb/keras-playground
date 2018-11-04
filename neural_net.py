from keras.models import Sequential
from keras.layers import Dense

import numpy as np

model = Sequential()

model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

# model.summary()

x_train =  np.array([
    [1, 2, 3, 4],
    [4, 6, 1, 2],
    [10, 9, 10, 11],
    [10, 12, 9, 13],
    [99, 100, 101, 102],
    [105, 111, 109, 102],
    [3, 7, 4, 1],
    [1, 8, 3, 7],
    [12, 15, 11, 9],
    [110, 99, 105, 101],
    [97, 101, 100, 105]
])

y_train = np.array([
    [2.5],
    [3.25],
    [10.0],
    [11.0],
    [100.5],
    [106.75]
])

perm = np.random.permutation(y_train.size)
x_train = x_train[perm]
y_train = y_train[perm]

x_val = np.array([
    [1.5, 4, 3, 2.5],
    [10, 14, 11.5, 12],
    [111, 99, 105, 107]
])

y_val = np.array([
    [2.75],
    [11.875],
    [105.5]
])

model.fit(
    x_train,
    y_train,
    batch_size=2,
    epochs=100,
    verbose=1,
    validation_data=(x_val, y_val)
)

model.save('mean_network.h5')

# x_predict = np.array([
#     [1.5, 2, 3.5, 4],
#     [13, 11, 9, 14],
#     [102, 98.5, 102.5, 100]
# ])

# output = model.predict(x_predict)

# print("")
# print("Expected: 2.75, 11.75, 100.75")
# print("Actual: ", output)

# x_test = np.array([
#     [2, 5, 4.5, 1],
#     [9, 16, 11, 10.5],
#     [100, 95, 99, 102]
# ])

# y_test =  np.array([
#     [3.125],
#     [11.625],
#     [99.0]
# ])

# output = model.evaluate(x_test, y_test)

# print("")
# print("=== Evaluation ===")
# print(model.metrics_names)
# print(output)