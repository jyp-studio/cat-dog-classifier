import tensorflow as tf
import pathlib


class DataLoader:
    def __init__(self, batch_size, img_size, train_path, val_path) -> None:
        self.batch_size = batch_size
        self.img_size = img_size
        self.load_dataset(train_path, val_path)

    def load_dataset(self, train_path, val_path):
        # set up path object
        train_dir = pathlib.Path(train_path)
        val_dir = pathlib.Path(val_path)

        # load training dataset
        print("load training set")
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size,
        )

        # load validation dataset
        print("load validation set")
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size,
        )

        # define class names
        self.class_names = self.train_ds.class_names

    def get_class_names(self):
        return self.class_names

    def normalize(self):
        # normalize image channel from [0, 255] to [0, 1]
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))

    def autotune(self, autotune):
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=autotune)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=autotune)


if __name__ == "__main__":
    """load dataset"""
    # define parameters
    batch_size = 32
    img_height = 224
    img_width = 224

    train_path = "./training_dataset"
    val_path = "./validation_dataset"

    # prefetch
    autotune = tf.data.AUTOTUNE

    data = DataLoader(batch_size, (img_height, img_width), train_path, val_path)
    print(data.get_class_names())
    data.normalize()
    data.autotune(autotune)
