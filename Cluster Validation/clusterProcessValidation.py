# %%
import random
import re

import pymongo

from ReadImages import readImage as rI
from ShowProcess import showProcesses as sP
from ImageProcessing import indAnalysis as inA
from ImageProcessing import imageOperations as iO
# %%
URI = 'mongodb://validationUser:85d4s32D2%23diA@idenmon.zapto.org:888/findValidation?authSource=findValidation'
CLIENT = pymongo.MongoClient(URI)
imagesCollection = CLIENT.findValidation.images
regx = re.compile("^102")
query = {'fileName': regx}
# %%
images = rI.customQuery(imagesCollection, query, limit=1)
random.shuffle(images)
# %%
for image in images:
    _, _, _, notMarkers = sP.showClusterProcess(image['file'], 3, 2, (10, 9), True)
    for marker in notMarkers:
        analysis = inA.deepQuadrantAnalysis(iO.notOperation(marker))
# %%
