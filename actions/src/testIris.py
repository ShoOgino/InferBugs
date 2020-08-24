from sklearn import datasets
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np

iris = datasets.load_iris()
X = iris.data
T = iris.target
T = np_utils.to_categorical(T)

np.random.seed(0) # 乱数初期化を固定値に
xTrain, xTest, yTrain, yTest = train_test_split(X, T, train_size=0.8, test_size=0.2)
len(xTrain), len(xTest), len(yTrain), len(yTest) # サイズ表示
print(xTrain)
model = Sequential()
model.add(Dense(input_dim=4, units=3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))
model.fit(xTrain, yTrain, epochs=50, batch_size=10)
Y=model.predict_classes(xTest, batch_size=10)
_, T_index = np.where(yTest > 0)
print(Y)
print(T_index)
