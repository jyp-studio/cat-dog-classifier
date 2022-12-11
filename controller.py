from PyQt6 import QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog
from gui import Ui_MainWindow

from PIL import Image
import cv2
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

from resnet50 import build_model


class Controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.setup_dataset()

        """model setting"""
        input_shape = (224, 224, 3)
        lr = 0.0001
        momentum = 0.001
        weights = "./model/resnet_v1.h5"

        optimizer = [
            tf.keras.optimizers.Adam(learning_rate=lr),
            tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
        ]

        loss = [
            tf.keras.losses.BinaryFocalCrossentropy(),
            tf.keras.losses.BinaryCrossentropy(),
        ]

        """load model"""
        self.model = build_model(input_shape)
        self.model.compile(optimizer=optimizer[0], loss=loss[0], metrics=["accuracy"])
        self.model.load_weights(weights)

    def setup_control(self):
        self.ui.btnLoad.clicked.connect(self.open_file)
        self.ui.btn5_1.clicked.connect(self.show5_1)
        self.ui.btn5_2.clicked.connect(self.show5_2)
        self.ui.btn5_3.clicked.connect(self.show5_3)
        self.ui.btn5_4.clicked.connect(self.show5_4)
        self.ui.btn5_5.clicked.connect(self.show5_5)

    def setup_dataset(self):
        self.batch_size = 32
        self.img_height = 224
        self.img_width = 224

        inference_path = "./inference_dataset"
        self.inference_dir = pathlib.Path(inference_path)
        self.inference_ds = tf.keras.utils.image_dataset_from_directory(
            self.inference_dir,
            seed=121,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )
        self.classes = self.inference_ds.class_names

    def open_file(self):
        self.filename = ""
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.ui.labelPredict.setText("")
        if self.filename != "":
            self.display_img()

    def display_img(self):
        self.img = cv2.imread(self.filename)
        self.img = cv2.resize(self.img, (224, 224), interpolation=cv2.INTER_AREA)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QtGui.QImage(
            self.img, width, height, bytesPerline, QtGui.QImage.Format.Format_RGB888
        ).rgbSwapped()
        self.ui.labelImg.setPixmap(QtGui.QPixmap.fromImage(self.qimg))

    def show5_1(self):
        """show images"""
        plt.figure()
        for images, labels in self.inference_ds.take(1):
            for i in range(2):
                plt.subplot(1, 2, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.classes[labels[i]])
                plt.axis("off")
        plt.show()

    def show5_2(self):
        """show distribution"""
        img = Image.open("./distribution.jpg")
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def show5_3(self):
        """show model structure"""
        print(self.model.summary())

    def show5_4(self):
        """show comparison"""
        img = Image.open("./accuracy_comparison.jpg")
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def show5_5(self):
        """inference"""
        theshold = 0.5

        img = self.img.reshape(1, 224, 224, 3)
        out = self.model.predict_step(img)
        prediction = "Cat" if out < theshold else "Dog"
        self.ui.labelPredict.setText(f"Prediction: {prediction}")
