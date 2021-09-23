import numpy as np 
import os 
import matplotlib.pyplot as plt 
from PIL import Image
from numpy.lib.npyio import load

BASE_PATH = "/localdata/helsinki_deblur/"

def load_data(font, step):
    assert font in ["Times", "Verdana"], "Font has to be either Times or Verdana"
    assert step in np.arange(20), "Step has to an integer between 0 and 19"

    cam1_path = os.path.join(*[BASE_PATH, "step" + str(step), font, "CAM01"])
    cam2_path = os.path.join(*[BASE_PATH, "step" + str(step), font, "CAM02"])

    x_images = []
    y_images = []
    for f in os.listdir(cam1_path):
        if f.endswith("tif") and "sample" in f:
            x = np.array(Image.open(os.path.join(cam1_path, f))) # not blurry
            y = np.array(Image.open(os.path.join(cam2_path, f))) # blurry

            x_images.append(x)
            y_images.append(y)

    return np.asarray(x_images), np.asarray(y_images)

def load_calibration_images(step, font="Times"):
    # does the font matter for the calibration image? We have a calibration image both in "Times" and in "Verdana"

    # CAM01: not blurry
    # CAM02: blurry 
    cam1_path = os.path.join(*[BASE_PATH, "step" + str(step), font, "CAM01"])
    cam2_path = os.path.join(*[BASE_PATH, "step" + str(step), font, "CAM02"])

    x_images = []
    y_images = []

    for img in ["focusStep_{}_PSF.tif".format(step), "focusStep_{}_LSF_Y.tif".format(step), "focusStep_{}_LSF_X.tif".format(step)]:
        
        x = np.array(Image.open(os.path.join(cam1_path, img))) # not blurry
        y = np.array(Image.open(os.path.join(cam2_path, img))) # blurry

        x_images.append(x)
        y_images.append(y)

    return np.asarray(x_images), np.asarray(y_images)


def load_data_with_text(font, step):
    assert font in ["Times", "Verdana"], "Font has to be either Times or Verdana"
    assert step in np.arange(20), "Step has to an integer between 0 and 19"

    cam1_path = os.path.join(*[BASE_PATH, "step" + str(step), font, "CAM01"])
    cam2_path = os.path.join(*[BASE_PATH, "step" + str(step), font, "CAM02"])

    x_images = []
    y_images = []
    true_text = []
    for f in os.listdir(cam1_path):
        if f.endswith("tif") and "sample" in f:
            x = np.array(Image.open(os.path.join(cam1_path, f))) # not blurry
            y = np.array(Image.open(os.path.join(cam2_path, f))) # blurry

            x_images.append(x)
            y_images.append(y)

            with open(os.path.join(cam1_path, f.split(".")[0] + ".txt"), 'r') as f:
                trueText = f.readlines()

            # remome \n character
            trueText = [text.rstrip() for text in trueText]
            true_text.append(trueText[1]) # trueText , only output middle line
            

    return np.asarray(x_images), np.asarray(y_images), true_text





if __name__ == "__main__":
    fonts = ["Times", "Verdana"]
    steps = np.arange(10)

    for font in fonts:
        for step in steps:
            X, Y = load_data(font, step)
            print("Font: ", font, " || Step: ", step, " \n Shape of X: ", X.shape, " \t Shape of Y: ", Y.shape)

            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))

            ax1.imshow(X[0, :, :], cmap="gray")
            ax1.set_title("Camera 1")

            ax2.imshow(Y[0, :, :], cmap="gray")
            ax2.set_title("Camera 2")

            plt.show()
