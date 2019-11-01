import os

import cv2
from matplotlib import pyplot as plt

# Scripts para leer y procesar imagen
print(os.getcwd())
workingPath = os.getcwd()
scriptsPath = '../Golden Master (AS IS)'
os.chdir(scriptsPath)
os.chdir(workingPath)


def showImages(imagesList):
    for image in imagesList:
        plt.imshow(image)
        plt.show()


# %%
imagesFolderPython = './clusterImagesPython/'
imagesFolderJava = './clusterImagesJava/'

imagesPython = os.listdir(imagesFolderPython)
imagesJava = os.listdir(imagesFolderJava)
# %%
for (imageNamePython, imageNameJava) in zip(imagesPython, imagesJava):
    # Python
    imagePathPython = os.path.join(imagesFolderPython, imageNamePython)
    imagePython = cv2.imread(imagePathPython)
    # Java
    imagePathJava = os.path.join(imagesFolderJava, imageNameJava)
    imageJava = cv2.imread(imagePathJava)
    # Matrix substraction
    substracted = imageJava/imagePython
    # Visual Comparison
    plt.subplot(131)
    plt.imshow(imagePython)
    plt.title('Python')
    plt.subplot(132)
    plt.imshow(imageJava)
    plt.title('Java')
    plt.subplot(133)
    plt.imshow(substracted)
    plt.title('Division')
    plt.show()