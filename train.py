import tensorflow as tf

from prepare_data import *
from resnet50 import *

if __name__ == "__main__":
    """setting"""
    # define parameters
    epochs = 2
    batch_size = 4
    img_height = 224
    img_width = 224
    channels = 3
    input_shape = (img_height, img_width, channels)

    train_path = "./training_dataset"
    val_path = "./validation_dataset"

    # prefetch
    autotune = tf.data.AUTOTUNE

    """load dataset"""
    data = DataLoader(batch_size, (img_height, img_width), train_path, val_path)
    data.normalize()
    data.autotune(autotune)

    """load model"""
    resnet50 = build_model(input_shape)
    print(resnet50.summary())

    """Hyper Parameter & learning scheduling"""
    lr = 0.0001
    momentum = 0.001
    version = 1
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", min_delta=0.001, mode="min", patience=7, verbose=1
    )
    csv_filename = [f"resnet50_v{version}.log"]
    csv = tf.keras.callbacks.CSVLogger(csv_filename[0])

    save_model = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"model/resnet_v{version}.h5",
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", min_delta=0.001, mode="min", patience=5, factor=0.1, verbose=1
    )
    optimizer = [
        tf.keras.optimizers.Adam(learning_rate=lr),
        tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
    ]
    loss = [
        tf.keras.losses.BinaryFocalCrossentropy(),
        tf.keras.losses.BinaryCrossentropy(),
    ]
    metrics = ["accuracy"]
    callbacks = [csv, early_stop, reduce_lr, save_model]

    """training execution"""
    resnet50.compile(optimizer=optimizer[0], loss=loss[0], metrics=metrics)
    resnet50.fit(
        data.train_ds,
        validation_data=data.val_ds,
        # batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
    )
