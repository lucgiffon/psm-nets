from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Layer
import tempfile
import pathlib
import keras

class CustomIdentity(Layer):
    def __init__(self, useless_arg, *args, **kwargs):
        self.useless_arg = useless_arg
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        return inputs

    def build(self, input_shape):
        super().build(input_shape=input_shape)

    def get_config(self):
        base_cfg = super().get_config()
        cfg = {
            "useless_arg": self.useless_arg
        }
        base_cfg.update(cfg)
        return base_cfg

    def compute_output_shape(self, input_shape):
        return input_shape


model = Sequential()
model.add(Conv2D(6, (5, 5), padding='valid', activation='relu', input_shape=(28, 28, 3)))
model.add(Flatten(name='flatten'))
model.add(Dense(1024, use_bias = True, name='fc1'))
model.add(CustomIdentity(useless_arg=10, name="useless_custom"))
model.add(Activation('softmax'))

model.compile(optimizer="adam", loss="mse")

with tempfile.TemporaryDirectory() as tmpdir:
    path_tmp = pathlib.Path(tmpdir) / "tmp.h5"
    model.save(str(path_tmp))
    keras.models.load_model(str(path_tmp), custom_objects={
        "CustomIdentity": CustomIdentity
    })