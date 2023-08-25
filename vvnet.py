from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Input, Layer, Dropout, UpSampling3D, concatenate

INPUT_SHAPE = (128, 256, 256, 3)


class Conv2Plus1D(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int | tuple[int],
        *,
        padding: str = "valid",
        activation: str = None,
        kernel_initializer: str = "glorot_uniform"
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        self.seqeuence = Sequential(
            [
                Conv3D(
                    filters,
                    (1, *kernel_size[1:]),
                    padding=padding,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                ),
                Conv3D(
                    filters,
                    (kernel_size[0], 1, 1),
                    padding=padding,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                ),
            ]
        )

    def __call__(self, x):
        return self.seqeuence(x)


def VVNet(pretrained_weights=None):
    in1 = Input(INPUT_SHAPE)

    conv1 = Conv2Plus1D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(in1)
    conv1 = Conv2Plus1D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv2Plus1D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = Conv2Plus1D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv2Plus1D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = Conv2Plus1D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(drop3)

    conv4 = Conv2Plus1D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = Conv2Plus1D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    drop4 = Dropout(0.5)(conv4)

    up5 = UpSampling3D(size=(2, 2, 2))(drop4)
    up5 = Conv2Plus1D(256, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up5)
    merge5 = concatenate([up5, drop3], axis=4)
    conv5 = Conv2Plus1D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge5)
    conv5 = Conv2Plus1D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)

    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    up6 = Conv2Plus1D(128, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up6)
    merge6 = concatenate([up6, conv2], axis=4)
    conv6 = Conv2Plus1D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = Conv2Plus1D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    up7 = Conv2Plus1D(64, 2, activation="relu", padding="same", kernel_initializer="he_normal")(up7)
    merge7 = concatenate([up7, conv1], axis=4)
    conv7 = Conv2Plus1D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = Conv2Plus1D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)
    conv8 = Conv2Plus1D(2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)
    conv9 = Conv2Plus1D(1, 1, activation="sigmoid")(conv8)

    model = Model(inputs=in1, outputs=conv9)
    model.compile(optimizer=Adam(learning_rate=5e-4), loss=BinaryCrossentropy())

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
