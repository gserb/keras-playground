import numpy as np

from keras.models import load_model
from keras.utils.np_utils import to_categorical

model = load_model('iris.h5')

data = np.genfromtxt('iris.csv', delimiter=',')

x_train = data[1:, :4]
y_train = to_categorical(data[1:, 4])

perm = np.random.permutation(y_train.shape[0])
x_train = x_train[perm]
y_train = y_train[perm]

model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_split=0.2
)