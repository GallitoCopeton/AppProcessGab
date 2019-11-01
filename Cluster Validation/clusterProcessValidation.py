#%%
from ImageFunctions.ShowProcess import showProcesses as sP
import os
import random
import re

import pymongo
import cv2
import numpy as np
from matplotlib import pyplot as plt

from ImageFunctions.ImageProcessing import perspective as pPe
from ImageFunctions.ImageProcessing import preProcessing as pP
from ImageFunctions.ImageProcessing import sorts as srt
from ImageFunctions.ImageProcessing import indAnalysis as inA
from ImageFunctions.AppProcess.MarkerProcess.markerProcess import clusteringProcess
from ImageFunctions.AppProcess.CroppingProcess import croppingProcess as crP
from ImageFunctions.ReadImages import readImage as rI


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def makeFolder(folderName):
    if not os.path.isdir(folderName):
        os.mkdir(folderName)


URI = 'mongodb://validationUser:85d4s32D2%23diA@idenmon.zapto.org:888/findValidation?authSource=findValidation'
CLIENT = pymongo.MongoClient(URI)
imagesCollection = CLIENT.findValidation.images
allResultsFolder = './clusterValidation'
makeFolder(allResultsFolder)
qrsAlreadySaved = os.listdir(allResultsFolder)
regx = re.compile("^102")
query = {'fileName': regx}
#query = {}
images = rI.readManyCustomQueryDetails(imagesCollection, limit=300)
random.shuffle(images)


# %%

# sP.showOldProcess(images[0]['image'])
for image in images:
    makeFolder(allResultsFolder+'/'+image['qr'])
    m = sP.showClusterProcess(image['image'], 30, 3, (8, 9), True)
