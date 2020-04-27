import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten

mod = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights="imagenet", input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=100)
# mod3 = keras.applications.densenet.DenseNet121(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=100)

top_model = Sequential()
top_model.add(Flatten())
top_model.add(Dense(256,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1,activation='sigmoid'))
mod2 = Model(inputs=mod.input, outputs=top_model(mod.output))

print(mod.summary())
print(mod2.summary())


