# %%
import random
import re

import pymongo

from ImageFunctions.ReadImages import readImage as rI
from ImageFunctions.ShowProcess import showProcesses as sP
# %%
URI = 'mongodb://validationUser:85d4s32D2%23diA@idenmon.zapto.org:888/findValidation?authSource=findValidation'
CLIENT = pymongo.MongoClient(URI)
imagesCollection = CLIENT.findValidation.images
regx = re.compile("^102")
query = {'fileName': regx}
# %%
images = rI.readManyCustomQueryDetails(imagesCollection, limit=2)
random.shuffle(images)
# %%
for image in images:
    m = sP.showClusterProcess(image['image'], 30, 3, (10, 9), True)
# %%
