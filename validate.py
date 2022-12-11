import tensorflow as tf
import pathlib
from resnet50 import *
import numpy as np

"""inference data"""
img_size = (224, 224)
batch_size = 20
inf_path = "./inference_dataset"
inf_dir = pathlib.Path(inf_path)
inf_ds = tf.keras.utils.image_dataset_from_directory(
    inf_dir,
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
)
classes = ["Cat", "Dog"]
x = []
y = []
for images, labels in inf_ds.take(len(inf_ds)):
    for i in range(len(images)):
        x.append(images[i])
        y.append(labels[i])
x = np.array(x)
y = np.array(y)

input_shape = (224, 224, 3)
lr = 0.0001
momentum = 0.001
weights = "./model/resnet_v1.h5"
net = build_model(input_shape=(224, 224, 3))
net.load_weights(weights)

optimizer = [
    tf.keras.optimizers.Adam(learning_rate=lr),
    tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
]
loss = [
    tf.keras.losses.BinaryFocalCrossentropy(),
    tf.keras.losses.BinaryCrossentropy(),
]

"""training execution"""
net.compile(optimizer=optimizer[0], loss=loss[0], metrics=["accuracy"])
img = x[0].reshape(1, 224, 224, 3)
result = net.predict_step(img)
print("Cat" if result < 0.5 else "Dog")
# result = net.evaluate(inf_ds)
# predicts = []
# for value in result:
#     if value < 0.5:
#         predicts.append("Cat")
#     else:
#         predicts.append("Dog")
# print(predicts)
