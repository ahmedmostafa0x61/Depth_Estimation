"""Desktop app to compute depth map for a selected image.
"""
import os
import glob
from shutil import copy2
import torch
import utils
import cv2
import timm

from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# Desktop Libraries

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType

import sys

# from app_ui import Ui_MainWindow as Program


# Or use the following
Program, _ = loadUiType('app_ui.ui')


class MainApp(QMainWindow, Program):
    def __init__(self):
        super(MainApp, self).__init__()
        QMainWindow.__init__(self)
        self.setupUi(self)
        # self.model_initialization()
        self.showMaximized()
        self.user_action()
        self.button_shortcuts()

    def button_shortcuts(self):
        self.select_pb.setShortcut("Ctrl+o")
        self.convert_pb.setShortcut("Ctrl+r")
        self.save_pb.setShortcut("Ctrl+s")

    def user_action(self):
        self.select_pb.clicked.connect(self.select)
        self.convert_pb.clicked.connect(self.write_depth_ori)
        self.save_pb.clicked.connect(self.save_image)
        self.select_acc.currentIndexChanged.connect(self.model_initialization)

    def button_activate(self, button):
        if button == 'convert':
            self.convert_pb.setEnabled(True)
        elif button == 'save':
            self.save_pb.setEnabled(True)

    def select(self):
        """
        open a window to select an image
        """
        global imagePath
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        imagePath = fname[0]
        pixmap = QPixmap(imagePath)
        width = self.img_before.width()  # Box width
        height = self.img_before.height()  # Box height
        self.img_before.setPixmap(pixmap.scaled(width, height, Qt.KeepAspectRatio))
        self.button_activate('convert')

    def model_initialization(self):
        global transform, device, model, optimize

        # initialization
        optimize = True
        model_type = "None"
        model_path = "None"
        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if self.select_acc.currentText() == 'Option 1':
            """DPT-large model"""
            model_type = "dpt_large"
            model_path = "weights/dpt_large-midas-2f21e586.pt"
        elif self.select_acc.currentText() == 'Option 2':
            '''midas v2.1 small '''
            model_type = "midas_v21_small"
            model_path = "weights/midas_v21_small-70d6b9c8.pt"

        # select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load network
        if model_type == "dpt_large":  # DPT-Large
            model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "midas_v21_small":
            model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                                   non_negative=True, blocks={'expand': True})
            net_w, net_h = 256, 256
            resize_mode = "upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        else:
            print(f"model_type '{model_type}' not implemented, use: --model_type large")
            assert False

        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        model.eval()

        if optimize:
            # rand_example = torch.rand(1, 3, net_h, net_w)
            # model(rand_example)
            # traced_script_module = torch.jit.trace(model, rand_example)
            # model = traced_script_module

            if device == torch.device("cuda"):
                model = model.to(memory_format=torch.channels_last)
                model = model.half()

        model.to(device)

    def convert(self):
        """
        Build the model and Convert the image to its Depth
        """

        print("start processing")

        # input
        img = utils.read_image(imagePath)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                    .squeeze()
                    .cpu()
                    .numpy()
            )

        return prediction

    def write_depth_ori(self):
        global pic_path
        prediction = self.convert()
        # to avoid dir is already exists
        try:
            os.mkdir('records')
        except FileExistsError:
            pass

        # Saving the Depth map into an image

        # get current dir
        current_path = os.getcwd()
        record_path = os.path.join(current_path, 'records')
        # get the last number in the output dir
        # to set the next number to the image to be saved
        converted_dir = len(os.listdir(record_path)) + 1
        output_path = os.path.join(record_path, str(converted_dir))
        # make a new dir to hold both the original and the converted image
        try:
            os.mkdir(output_path)
        except FileExistsError:
            converted_dir += 1
            output_path = os.path.join(record_path, str(converted_dir))

        # save the converted Image
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(imagePath))[0]
        )
        pic_path = utils.write_depth(filename, prediction, bits=2)

        # Copy the original Image
        copy2(imagePath, filename + '.png')

        # Show the Depth Map into QWidget
        qpic = QPixmap(pic_path)
        width = self.img_after.width()  # Box width
        height = self.img_after.height()  # Box height
        self.img_after.setPixmap(qpic.scaled(width, height, Qt.KeepAspectRatio))
        self.button_activate('save')

    def save_image(self):

        name = QFileDialog.getSaveFileName(self, 'Save File')[0]
        print(name + '.png')
        copy2(pic_path, name + '.png')


if __name__ == "__main__":
    # Run Desktop app
    app = QApplication(sys.argv)
    window = MainApp()
    window.setWindowTitle('Arabesque')
    window.setWindowIcon(QIcon('layers.ico'))
    window.show()
    app.exec_()
