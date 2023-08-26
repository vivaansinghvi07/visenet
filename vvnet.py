from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential, Model
from tensorflow.keras.backend import mean, binary_crossentropy
from tensorflow.keras.layers import (
    Conv3D,
    MaxPooling3D,
    Input,
    Layer,
    Dropout,
    UpSampling3D,
    concatenate,
    BatchNormalization,
    Activation,
)

INPUT_SHAPE = (32, 128, 128, 3)


# https://stackoverflow.com/questions/73352641/my-unet-model-produces-an-all-gray-picture
def weighted_bincrossentropy(true, pred, weight_zero=0.10, weight_one=1):
    """
    Calculates weighted binary cross entropy. The weights are fixed.

    This can be useful for unbalanced catagories.

    Adjust the weights here depending on what is required.

    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives
        will be penalize 10 times as much as false negatives.
    """

    # calculate the binary cross entropy
    bin_crossentropy = binary_crossentropy(true, pred)

    # apply the weights
    weights = true * weight_one + (1.0 - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return mean(weighted_bin_crossentropy)


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


def VVNet(*, vdepth=3, pretrained_weights=None, learning_rate=5e-4):
    """
    Start at 64, multiply by factor of two per iteration of the downwards step.
    Then, go backwards, ending at 1.

    With depth 4:
    64 >> 128 >> 256 >> 512 >> 256 >> 128 >> 64 >> 1

    With depth 2:
    64 >> 128 >> 64 >> 1
    """

    input_layer = Input(INPUT_SHAPE)
    concat_layers = []

    # load downwards steps for the model
    for i in range(vdepth - 1):
        # add convolutional steps
        if i == 0:
            model_layer = Conv2Plus1D(64 * (2**i), 3, padding="same", kernel_initializer="he_normal")(input_layer)
        else:
            model_layer = Conv2Plus1D(64 * (2**i), 3, padding="same", kernel_initializer="he_normal")(model_layer)
        model_layer = Conv2Plus1D(64 * (2**i), 3, padding="same", kernel_initializer="he_normal")(model_layer)
        model_layer = BatchNormalization()(model_layer)
        model_layer = Activation("relu")(model_layer)

        # add an extra dropout layer if right before end
        if i == vdepth - 2:
            model_layer = Dropout(0.5)(model_layer)
        concat_layers.append(model_layer)

        # perform pooling and go downwards
        model_layer = MaxPooling3D(pool_size=(2, 2, 2))(model_layer)

    # add the bottom step, when 3, vdepth should be 64 * 4 = 256
    model_layer = Conv2Plus1D(64 * (2 ** (vdepth - 1)), 3, padding="same", kernel_initializer="he_normal")(model_layer)
    model_layer = Conv2Plus1D(64 * (2 ** (vdepth - 1)), 3, padding="same", kernel_initializer="he_normal")(model_layer)
    model_layer = BatchNormalization()(model_layer)
    model_layer = Activation("relu")(model_layer)
    model_layer = Dropout(0.5)(model_layer)

    # upwards steps
    for i in range(vdepth - 2, -1, -1):
        # perform the upwards step
        model_layer = UpSampling3D(size=(2, 2, 2))(model_layer)
        model_layer = Conv2Plus1D(64 * (2**i), 2, padding="same", kernel_initializer="he_normal")(model_layer)
        model_layer = BatchNormalization()(model_layer)
        model_layer = Activation("relu")(model_layer)
        model_layer = concatenate([model_layer, concat_layers.pop()], axis=4)

        # convole further
        model_layer = Conv2Plus1D(64 * (2**i), 3, padding="same", kernel_initializer="he_normal")(model_layer)
        model_layer = Conv2Plus1D(64 * (2**i), 3, padding="same", kernel_initializer="he_normal")(model_layer)
        model_layer = BatchNormalization()(model_layer)
        model_layer = Activation("relu")(model_layer)

    # finally, get everything in the output format
    model_layer = Conv2Plus1D(2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(model_layer)
    model_layer = Conv2Plus1D(1, 1, activation="sigmoid")(model_layer)

    model = Model(inputs=input_layer, outputs=model_layer)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=weighted_bincrossentropy, metrics=["accuracy"])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

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
    model.compile(optimizer=Adam(learning_rate=5e-4), loss=weighted_bincrossentropy)

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
