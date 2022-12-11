from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications.resnet import ResNet50


def build_model(input_shape):
    net = ResNet50(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
    )
    x = net.output
    # add fc layers
    x = Flatten()(x)
    # add Dense layer and get probs by softmax
    output_layer = Dense(1, activation="sigmoid", name="sigmoid")(x)
    # define freeze and train layers
    net_final = Model(inputs=net.input, outputs=output_layer)

    return net_final


if __name__ == "__main__":
    input_shape = (224, 224, 3)
    net = build_model(input_shape)
    # print model
    print(net.summary())
