from keras.models import Sequential
from keras.layers import Dense

import numpy as np

model = Sequential()


# 10.5, 5, 9.5, 12 => low

# < 50 = low
# > 50 = high

model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 0 = low
# 1 = high

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy',
    metrics=['accuracy']
)

data = np.genfromtxt('high_low.csv', delimiter=',')

x_train = data[1:, :4]
y_train = data[1:, 4]

model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_split=0.2
)

x_predict = np.array([
    [10, 25, 14, 9],
    [102, 100, 75, 90]
])

output = model.predict_classes(x_predict)
print("")
print(output)