# %%
import readImage
import indAnalysis as inA
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pymongo
import seaborn as sns
from matplotlib import pyplot as plt

# Scripts para leer y procesar imagen
sys.path.insert(0, '../Golden Master (AS IS)')
try:
    os.chdir(os.path.join(os.getcwd(), 'Helper notebooks'))
    print(os.getcwd())
except:
    print(os.getcwd())


# %% Read all the images from the database prodLaboratorio in collection markerTotals that have diagnosis
MONGO_URL = 'mongodb://findOnlyReadUser:RojutuNHqy@idenmon.zapto.org:888/?authSource=prodLaboratorio'
client = pymongo.MongoClient(MONGO_URL)
db = client.prodLaboratorio
markerTotals = db.markerTotals
markers = markerTotals.find({
    'diagnostic': {'$ne': None},
    'marker': {'$ne': 'P24'}
})
count = 1
mask = inA.readMask()
stats = ['area', 'diagnostic']
markerNames = ['ESAT6',
               'CFP10',
               'RV1681']
quadrants = ['Q1', 'Q2', 'Q3', 'Q4']
markersDataframes = inA.createQuadrantDataframes(
    markerNames, quadrants, stats, None)
ESAT6 = markersDataframes[0]
CFP10 = markersDataframes[1]
RV1681 = markersDataframes[2]
ESAT6_Q1 = ESAT6['Q1']
ESAT6_Q2 = ESAT6['Q2']
ESAT6_Q3 = ESAT6['Q3']
ESAT6_Q4 = ESAT6['Q4']
CFP10_Q1 = CFP10['Q1']
CFP10_Q2 = CFP10['Q2']
CFP10_Q3 = CFP10['Q3']
CFP10_Q4 = CFP10['Q4']
RV1681_Q1 = RV1681['Q1']
RV1681_Q2 = RV1681['Q2']
RV1681_Q3 = RV1681['Q3']
RV1681_Q4 = RV1681['Q4']
skippedImage = 0
ESAT6_Total, CFP10_Total, RV1681_Total = inA.createQuadrantDataframes(
    markerNames, markerNames, stats, None)
ESAT6_Total = ESAT6_Total['ESAT6']
CFP10_Total = CFP10_Total['CFP10']
RV1681_Total = RV1681_Total['RV1681']
# %%
for imageNumber, marker in enumerate(markers):

    imageBase64 = marker['image']
    originalImage = inA.resizeFixed(readImage.readb64(imageBase64))
    if count > 0:
        if(isinstance(originalImage, str)):
            print(originalImage)
            continue
        else:
            # Show original image
            #plt.imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))

            # Image preprocessing => START

            # Reshape image to have the RGB features (3) each as a column (-1)
            Z = originalImage.reshape((-1, 3))
            # Convert to np.float32
            Z = np.float32(Z)
            # Define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 3
            ret, label, centers = cv2.kmeans(
                Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # Now convert back into uint8 and make original image
            centers = np.uint8(centers)
            kMeansReconstructed = centers[label.flatten()]
            # kMeansReconstructedReshaped = cv2.GaussianBlur(
            #     kMeansReconstructed.reshape((originalImage.shape)), (5, 5), 0)
            kMeansReconstructedReshaped = kMeansReconstructed.reshape(
                originalImage.shape)
            # Show reconstructed image
            #plt.imshow(cv2.cvtColor(kMeansReconstructedReshaped, cv2.COLOR_BGR2RGB))
            # Process reconstructed image
            imgGray = cv2.cvtColor(
                kMeansReconstructedReshaped, cv2.COLOR_BGR2GRAY)
            _, imgBina = cv2.threshold(imgGray, 160, 255, cv2.THRESH_BINARY)
            # Show img binarized
            #plt.imshow(imgBina, 'gray')
            imgBinaMask = inA.andOperation(imgBina, mask)
            kernel = np.ones((1, 1), np.uint8)
            #imgBinaMaskInvEroDil = cv2.dilate(imgBinaMask, kernel, iterations=1)
            imgBinaMaskInvEroDil = cv2.morphologyEx(
                imgBinaMask, cv2.MORPH_OPEN, kernel)
            # Show blobs
            #plt.imshow(imgBinaMaskInvEroDil, 'gray')

            # Image preprocessing => FINISH

            # Dataframe creation => START
            skip = False
            try:
                pixelsPerQuadrant = inA.quadrantAreaAnalysis(
                    imgBinaMaskInvEroDil)
                totalPixels = inA.whitePixels(imgBinaMaskInvEroDil)
                tempDFs = []

                relevanceThresh = totalPixels/4
                trueThresh = 0
                for pixels in pixelsPerQuadrant:
                    if pixels >= relevanceThresh:
                        trueThresh += pixels
                    tempDF = pd.DataFrame.from_dict({
                        'area': [float(pixels)],
                        'diagnostic': [marker['diagnostic'].upper()]
                    })
                    tempDFs.append(tempDF)
                tempDfTotal = pd.DataFrame.from_dict({
                    'area': [float(totalPixels)],
                    'diagnostic': [marker['diagnostic'].upper()]
                })
            except:
                continue
            if skip:
                skippedImage += 1
                continue
            if marker['marker'] == 'ESAT6':
                ESAT6_Total = ESAT6_Total.append(tempDfTotal)
                ESAT6_Q1 = ESAT6_Q1.append(tempDFs[0])
                ESAT6_Q2 = ESAT6_Q2.append(tempDFs[1])
                ESAT6_Q3 = ESAT6_Q3.append(tempDFs[2])
                ESAT6_Q4 = ESAT6_Q4.append(tempDFs[3])
            elif marker['marker'] == 'CFP10':
                CFP10_Total = CFP10_Total.append(tempDfTotal)
                CFP10_Q1 = CFP10_Q1.append(tempDFs[0])
                CFP10_Q2 = CFP10_Q2.append(tempDFs[1])
                CFP10_Q3 = CFP10_Q3.append(tempDFs[2])
                CFP10_Q4 = CFP10_Q4.append(tempDFs[3])
            elif marker['marker'] == 'RV1681':
                RV1681_Total = RV1681_Total.append(tempDfTotal)
                RV1681_Q1 = RV1681_Q1.append(tempDFs[0])
                RV1681_Q2 = RV1681_Q2.append(tempDFs[1])
                RV1681_Q3 = RV1681_Q3.append(tempDFs[2])
                RV1681_Q4 = RV1681_Q4.append(tempDFs[3])
            # Dataframe creation => FINISH

            imageNumber += 1
            print(
                f'\nImágenes procesadas: {imageNumber}\nImágenes restantes: {2376-imageNumber}\nImagenes excluidas: {skippedImage}')
            print(marker['marker'])
            plt.show()
# %%
CFP10[('Q1', 'area')] = CFP10_Q1['area']
CFP10[('Q1', 'diagnostic')] = CFP10_Q1['diagnostic']
CFP10[('Q2', 'area')] = CFP10_Q2['area']
CFP10[('Q2', 'diagnostic')] = CFP10_Q2['diagnostic']
CFP10[('Q3', 'area')] = CFP10_Q3['area']
CFP10[('Q3', 'diagnostic')] = CFP10_Q3['diagnostic']
CFP10[('Q4', 'area')] = CFP10_Q4['area']
CFP10[('Q4', 'diagnostic')] = CFP10_Q4['diagnostic']
#####################################################
ESAT6[('Q1', 'area')] = ESAT6_Q1['area']
ESAT6[('Q1', 'diagnostic')] = ESAT6_Q1['diagnostic']
ESAT6[('Q2', 'area')] = ESAT6_Q2['area']
ESAT6[('Q2', 'diagnostic')] = ESAT6_Q2['diagnostic']
ESAT6[('Q3', 'area')] = ESAT6_Q3['area']
ESAT6[('Q3', 'diagnostic')] = ESAT6_Q3['diagnostic']
ESAT6[('Q4', 'area')] = ESAT6_Q4['area']
ESAT6[('Q4', 'diagnostic')] = ESAT6_Q4['diagnostic']
#####################################################
RV1681[('Q1', 'area')] = RV1681_Q1['area']
RV1681[('Q1', 'diagnostic')] = RV1681_Q1['diagnostic']
RV1681[('Q2', 'area')] = RV1681_Q2['area']
RV1681[('Q2', 'diagnostic')] = RV1681_Q2['diagnostic']
RV1681[('Q3', 'area')] = RV1681_Q3['area']
RV1681[('Q3', 'diagnostic')] = RV1681_Q3['diagnostic']
RV1681[('Q4', 'area')] = RV1681_Q4['area']
RV1681[('Q4', 'diagnostic')] = RV1681_Q4['diagnostic']
RV1681[('Q1', 'area')] = RV1681[('Q1', 'area')]
#####################################################

# %%
# POSITIVES QUADRANTS CFP
f, (hist1, hist2, hist3, hist4) = plt.subplots(
    4, sharex=True, gridspec_kw={"height_ratios": (.25, .25, .25, .25)})
f.suptitle('Positive quadrants CFP')
sns.distplot(CFP10[(CFP10[('Q1', 'diagnostic')] == 'P')][('Q1', 'area')],
             label='P', bins=100, kde=True, ax=hist1, color="plum")
sns.distplot(CFP10[(CFP10[('Q2', 'diagnostic')] == 'P')][('Q2', 'area')],
             label='P', bins=100, kde=True, ax=hist2, color="plum")
sns.distplot(CFP10[(CFP10[('Q3', 'diagnostic')] == 'P')][('Q3', 'area')],
             label='P', bins=100, kde=True, ax=hist3, color="plum")
sns.distplot(CFP10[(CFP10[('Q4', 'diagnostic')] == 'P')][('Q4', 'area')],
             label='P', bins=100, kde=True, ax=hist4, color="plum")
# NEGATIVES  QUADRANTS CFP
f, (hist1, hist2, hist3, hist4) = plt.subplots(
    4, sharex=True, gridspec_kw={"height_ratios": (.25, .25, .25, .25)})
f.suptitle('Negative quadrants CFP')
sns.distplot(CFP10[CFP10[('Q1', 'diagnostic')] == 'N'][('Q1', 'area')],
             label='N', bins=100, kde=True, ax=hist1, color="seagreen")
sns.distplot(CFP10[CFP10[('Q2', 'diagnostic')] == 'N'][('Q2', 'area')],
             label='N', bins=100, kde=True, ax=hist2, color="seagreen")
sns.distplot(CFP10[CFP10[('Q3', 'diagnostic')] == 'N'][('Q3', 'area')],
             label='N', bins=100, kde=True, ax=hist3, color="seagreen")
sns.distplot(CFP10[CFP10[('Q4', 'diagnostic')] == 'N'][('Q4', 'area')],
             label='N', bins=100, kde=True, ax=hist4, color="seagreen")
# Mean and Std positive markers
CFP10AreaPositiveMean_Q1 = CFP10[CFP10[(
    'Q1', 'diagnostic')] == 'P'][('Q1', 'area')].mean()
CFP10AreaPositiveStd_Q1 = CFP10[CFP10[(
    'Q1', 'diagnostic')] == 'P'][('Q1', 'area')].std()
CFP10AreaPositiveMean_Q2 = CFP10[CFP10[(
    'Q2', 'diagnostic')] == 'P'][('Q2', 'area')].mean()
CFP10AreaPositiveStd_Q2 = CFP10[CFP10[(
    'Q2', 'diagnostic')] == 'P'][('Q2', 'area')].std()
CFP10AreaPositiveMean_Q3 = CFP10[CFP10[(
    'Q3', 'diagnostic')] == 'P'][('Q3', 'area')].mean()
CFP10AreaPositiveStd_Q3 = CFP10[CFP10[(
    'Q3', 'diagnostic')] == 'P'][('Q3', 'area')].std()
CFP10AreaPositiveMean_Q4 = CFP10[CFP10[(
    'Q4', 'diagnostic')] == 'P'][('Q4', 'area')].mean()
CFP10AreaPositiveStd_Q4 = CFP10[CFP10[(
    'Q4', 'diagnostic')] == 'P'][('Q4', 'area')].std()

print(f'Area positive mean CFP10 Q1: {CFP10AreaPositiveMean_Q1}')
print(f'Std positive CFP10 Q1: {CFP10AreaPositiveStd_Q1}')
print(f'Area positive mean CFP10 Q2: {CFP10AreaPositiveMean_Q2}')
print(f'Std positive CFP10 Q2: {CFP10AreaPositiveStd_Q2}')
print(f'Area positive mean CFP10 Q3: {CFP10AreaPositiveMean_Q3}')
print(f'Std positive CFP10 Q3: {CFP10AreaPositiveStd_Q3}')
print(f'Area positive mean CFP10 Q4: {CFP10AreaPositiveMean_Q4}')
print(f'Std positive CFP10 Q4: {CFP10AreaPositiveStd_Q4}\n')

# Mean and Std negative markers
CFP10AreaNegativeMean_Q1 = CFP10[CFP10[(
    'Q1', 'diagnostic')] == 'N'][('Q1', 'area')].mean()
CFP10AreaNegativeStd_Q1 = CFP10[CFP10[(
    'Q1', 'diagnostic')] == 'N'][('Q1', 'area')].std()
CFP10AreaNegativeMean_Q2 = CFP10[CFP10[(
    'Q2', 'diagnostic')] == 'N'][('Q2', 'area')].mean()
CFP10AreaNegativeStd_Q2 = CFP10[CFP10[(
    'Q2', 'diagnostic')] == 'N'][('Q2', 'area')].std()
CFP10AreaNegativeMean_Q3 = CFP10[CFP10[(
    'Q3', 'diagnostic')] == 'N'][('Q3', 'area')].mean()
CFP10AreaNegativeStd_Q3 = CFP10[CFP10[(
    'Q3', 'diagnostic')] == 'N'][('Q3', 'area')].std()
CFP10AreaNegativeMean_Q4 = CFP10[CFP10[(
    'Q4', 'diagnostic')] == 'N'][('Q4', 'area')].mean()
CFP10AreaNegativeStd_Q4 = CFP10[CFP10[(
    'Q4', 'diagnostic')] == 'N'][('Q4', 'area')].std()

print(f'Area negative mean CFP10 Q1: {CFP10AreaNegativeMean_Q1}')
print(f'Std negative CFP10 Q1: {CFP10AreaNegativeStd_Q1}')
print(f'Area negative mean CFP10 Q2: {CFP10AreaNegativeMean_Q2}')
print(f'Std negative CFP10 Q2: {CFP10AreaNegativeStd_Q2}')
print(f'Area negative mean CFP10 Q3: {CFP10AreaNegativeMean_Q3}')
print(f'Std negative CFP10 Q3: {CFP10AreaNegativeStd_Q3}')
print(f'Area negative mean CFP10 Q4: {CFP10AreaNegativeMean_Q4}')
print(f'Std negative CFP10 Q4: {CFP10AreaNegativeStd_Q4}')

# %% POSITIVES QUADRANTS ESAT6
# POSITIVES QUADRANTS CFP
f, (hist1, hist2, hist3, hist4) = plt.subplots(
    4, sharex=True, gridspec_kw={"height_ratios": (.25, .25, .25, .25)})
f.suptitle('Positive quadrants CFP')
sns.distplot(ESAT6[(ESAT6[('Q1', 'diagnostic')] == 'P')][('Q1', 'area')],
             label='P', bins=100, kde=True, ax=hist1, color="plum")
sns.distplot(ESAT6[(ESAT6[('Q2', 'diagnostic')] == 'P')][('Q2', 'area')],
             label='P', bins=100, kde=True, ax=hist2, color="plum")
sns.distplot(ESAT6[(ESAT6[('Q3', 'diagnostic')] == 'P')][('Q3', 'area')],
             label='P', bins=100, kde=True, ax=hist3, color="plum")
sns.distplot(ESAT6[(ESAT6[('Q4', 'diagnostic')] == 'P')][('Q4', 'area')],
             label='P', bins=100, kde=True, ax=hist4, color="plum")
# NEGATIVES  QUADRANTS ESAT6
f, (hist1, hist2, hist3, hist4) = plt.subplots(
    4, sharex=True, gridspec_kw={"height_ratios": (.25, .25, .25, .25)})
f.suptitle('Negative quadrants ESAT6')
sns.distplot(ESAT6[ESAT6[('Q1', 'diagnostic')] == 'N'][('Q1', 'area')],
             label='N', bins=100, kde=False, ax=hist1, color="seagreen")
sns.distplot(ESAT6[ESAT6[('Q2', 'diagnostic')] == 'N'][('Q2', 'area')],
             label='N', bins=100, kde=False, ax=hist2, color="seagreen")
sns.distplot(ESAT6[ESAT6[('Q3', 'diagnostic')] == 'N'][('Q3', 'area')],
             label='N', bins=100, kde=False, ax=hist3, color="seagreen")
sns.distplot(ESAT6[ESAT6[('Q4', 'diagnostic')] == 'N'][('Q4', 'area')],
             label='N', bins=100, kde=False, ax=hist4, color="seagreen")

# Mean and Std positive markers
ESAT6AreaPositiveMean_Q1 = ESAT6[ESAT6[(
    'Q1', 'diagnostic')] == 'P'][('Q1', 'area')].mean()
ESAT6AreaPositiveStd_Q1 = ESAT6[ESAT6[(
    'Q1', 'diagnostic')] == 'P'][('Q1', 'area')].std()
ESAT6AreaPositiveMean_Q2 = ESAT6[ESAT6[(
    'Q2', 'diagnostic')] == 'P'][('Q2', 'area')].mean()
ESAT6AreaPositiveStd_Q2 = ESAT6[ESAT6[(
    'Q2', 'diagnostic')] == 'P'][('Q2', 'area')].std()
ESAT6AreaPositiveMean_Q3 = ESAT6[ESAT6[(
    'Q3', 'diagnostic')] == 'P'][('Q3', 'area')].mean()
ESAT6AreaPositiveStd_Q3 = ESAT6[ESAT6[(
    'Q3', 'diagnostic')] == 'P'][('Q3', 'area')].std()
ESAT6AreaPositiveMean_Q4 = ESAT6[ESAT6[(
    'Q4', 'diagnostic')] == 'P'][('Q4', 'area')].mean()
ESAT6AreaPositiveStd_Q4 = ESAT6[ESAT6[(
    'Q4', 'diagnostic')] == 'P'][('Q4', 'area')].std()

print(f'Area positive mean ESAT6 Q1: {ESAT6AreaPositiveMean_Q1}')
print(f'Std positive ESAT6 Q1: {ESAT6AreaPositiveStd_Q1}')
print(f'Area positive mean ESAT6 Q2: {ESAT6AreaPositiveMean_Q2}')
print(f'Std positive ESAT6 Q2: {ESAT6AreaPositiveStd_Q2}')
print(f'Area positive mean ESAT6 Q3: {ESAT6AreaPositiveMean_Q3}')
print(f'Std positive ESAT6 Q3: {ESAT6AreaPositiveStd_Q3}')
print(f'Area positive mean ESAT6 Q4: {ESAT6AreaPositiveMean_Q4}')
print(f'Std positive ESAT6 Q4: {ESAT6AreaPositiveStd_Q4}\n')

# Mean and Std negative markers
ESAT6AreaNegativeMean_Q1 = ESAT6[ESAT6[(
    'Q1', 'diagnostic')] == 'N'][('Q1', 'area')].mean()
ESAT6AreaNegativeStd_Q1 = ESAT6[ESAT6[(
    'Q1', 'diagnostic')] == 'N'][('Q1', 'area')].std()
ESAT6AreaNegativeMean_Q2 = ESAT6[ESAT6[(
    'Q2', 'diagnostic')] == 'N'][('Q2', 'area')].mean()
ESAT6AreaNegativeStd_Q2 = ESAT6[ESAT6[(
    'Q2', 'diagnostic')] == 'N'][('Q2', 'area')].std()
ESAT6AreaNegativeMean_Q3 = ESAT6[ESAT6[(
    'Q3', 'diagnostic')] == 'N'][('Q3', 'area')].mean()
ESAT6AreaNegativeStd_Q3 = ESAT6[ESAT6[(
    'Q3', 'diagnostic')] == 'N'][('Q3', 'area')].std()
ESAT6AreaNegativeMean_Q4 = ESAT6[ESAT6[(
    'Q4', 'diagnostic')] == 'N'][('Q4', 'area')].mean()
ESAT6AreaNegativeStd_Q4 = ESAT6[ESAT6[(
    'Q4', 'diagnostic')] == 'N'][('Q4', 'area')].std()

print(f'Area negative mean ESAT6 Q1: {ESAT6AreaNegativeMean_Q1}')
print(f'Std negative ESAT6 Q1: {ESAT6AreaNegativeStd_Q1}')
print(f'Area negative mean ESAT6 Q2: {ESAT6AreaNegativeMean_Q2}')
print(f'Std negative ESAT6 Q2: {ESAT6AreaNegativeStd_Q2}')
print(f'Area negative mean ESAT6 Q3: {ESAT6AreaNegativeMean_Q3}')
print(f'Std negative ESAT6 Q3: {ESAT6AreaNegativeStd_Q3}')
print(f'Area negative mean ESAT6 Q4: {ESAT6AreaNegativeMean_Q4}')
print(f'Std negative ESAT6 Q4: {ESAT6AreaNegativeStd_Q4}')
# %% POSITIVES QUADRANTS RV1681
# POSITIVES QUADRANTS CFP
f, (hist1, hist2, hist3, hist4) = plt.subplots(
    4, sharex=True, gridspec_kw={"height_ratios": (.25, .25, .25, .25)})
f.suptitle('Positive quadrants CFP')
sns.distplot(RV1681[(RV1681[('Q1', 'diagnostic')] == 'P')][('Q1', 'area')],
             label='P', bins=100, kde=True, ax=hist1, color="plum")
sns.distplot(RV1681[(RV1681[('Q2', 'diagnostic')] == 'P')][('Q2', 'area')],
             label='P', bins=100, kde=True, ax=hist2, color="plum")
sns.distplot(RV1681[(RV1681[('Q3', 'diagnostic')] == 'P')][('Q3', 'area')],
             label='P', bins=100, kde=True, ax=hist3, color="plum")
sns.distplot(RV1681[(RV1681[('Q4', 'diagnostic')] == 'P')][('Q4', 'area')],
             label='P', bins=100, kde=True, ax=hist4, color="plum")
# NEGATIVES  QUADRANTS RV1681
f, (hist1, hist2, hist3, hist4) = plt.subplots(
    4, sharex=True, gridspec_kw={"height_ratios": (.25, .25, .25, .25)})
f.suptitle('Negative quadrants RV1681')
sns.distplot(RV1681[RV1681[('Q1', 'diagnostic')] == 'N'][('Q1', 'area')],
             label='N', bins=100, kde=False, ax=hist1, color="seagreen")
sns.distplot(RV1681[RV1681[('Q2', 'diagnostic')] == 'N'][('Q2', 'area')],
             label='N', bins=100, kde=False, ax=hist2, color="seagreen")
sns.distplot(RV1681[RV1681[('Q3', 'diagnostic')] == 'N'][('Q3', 'area')],
             label='N', bins=100, kde=False, ax=hist3, color="seagreen")
sns.distplot(RV1681[RV1681[('Q4', 'diagnostic')] == 'N'][('Q4', 'area')],
             label='N', bins=100, kde=False, ax=hist4, color="seagreen")

# Mean and Std positive markers
RV1681AreaPositiveMean_Q1 = RV1681[RV1681[(
    'Q1', 'diagnostic')] == 'P'][('Q1', 'area')].mean()
RV1681AreaPositiveStd_Q1 = RV1681[RV1681[(
    'Q1', 'diagnostic')] == 'P'][('Q1', 'area')].std()
RV1681AreaPositiveMean_Q2 = RV1681[RV1681[(
    'Q2', 'diagnostic')] == 'P'][('Q2', 'area')].mean()
RV1681AreaPositiveStd_Q2 = RV1681[RV1681[(
    'Q2', 'diagnostic')] == 'P'][('Q2', 'area')].std()
RV1681AreaPositiveMean_Q3 = RV1681[RV1681[(
    'Q3', 'diagnostic')] == 'P'][('Q3', 'area')].mean()
RV1681AreaPositiveStd_Q3 = RV1681[RV1681[(
    'Q3', 'diagnostic')] == 'P'][('Q3', 'area')].std()
RV1681AreaPositiveMean_Q4 = RV1681[RV1681[(
    'Q4', 'diagnostic')] == 'P'][('Q4', 'area')].mean()
RV1681AreaPositiveStd_Q4 = RV1681[RV1681[(
    'Q4', 'diagnostic')] == 'P'][('Q4', 'area')].std()

print(f'Area positive mean RV1681 Q1: {RV1681AreaPositiveMean_Q1}')
print(f'Std positive RV1681 Q1: {RV1681AreaPositiveStd_Q1}')
print(f'Area positive mean RV1681 Q2: {RV1681AreaPositiveMean_Q2}')
print(f'Std positive RV1681 Q2: {RV1681AreaPositiveStd_Q2}')
print(f'Area positive mean RV1681 Q3: {RV1681AreaPositiveMean_Q3}')
print(f'Std positive RV1681 Q3: {RV1681AreaPositiveStd_Q3}')
print(f'Area positive mean RV1681 Q4: {RV1681AreaPositiveMean_Q4}')
print(f'Std positive RV1681 Q4: {RV1681AreaPositiveStd_Q4}\n')

# Mean and Std negative markers
RV1681AreaNegativeMean_Q1 = RV1681[RV1681[(
    'Q1', 'diagnostic')] == 'N'][('Q1', 'area')].mean()
RV1681AreaNegativeStd_Q1 = RV1681[RV1681[(
    'Q1', 'diagnostic')] == 'N'][('Q1', 'area')].std()
RV1681AreaNegativeMean_Q2 = RV1681[RV1681[(
    'Q2', 'diagnostic')] == 'N'][('Q2', 'area')].mean()
RV1681AreaNegativeStd_Q2 = RV1681[RV1681[(
    'Q2', 'diagnostic')] == 'N'][('Q2', 'area')].std()
RV1681AreaNegativeMean_Q3 = RV1681[RV1681[(
    'Q3', 'diagnostic')] == 'N'][('Q3', 'area')].mean()
RV1681AreaNegativeStd_Q3 = RV1681[RV1681[(
    'Q3', 'diagnostic')] == 'N'][('Q3', 'area')].std()
RV1681AreaNegativeMean_Q4 = RV1681[RV1681[(
    'Q4', 'diagnostic')] == 'N'][('Q4', 'area')].mean()
RV1681AreaNegativeStd_Q4 = RV1681[RV1681[(
    'Q4', 'diagnostic')] == 'N'][('Q4', 'area')].std()

print(f'Area negative mean RV1681 Q1: {RV1681AreaNegativeMean_Q1}')
print(f'Std negative RV1681 Q1: {RV1681AreaNegativeStd_Q1}')
print(f'Area negative mean RV1681 Q2: {RV1681AreaNegativeMean_Q2}')
print(f'Std negative RV1681 Q2: {RV1681AreaNegativeStd_Q2}')
print(f'Area negative mean RV1681 Q3: {RV1681AreaNegativeMean_Q3}')
print(f'Std negative RV1681 Q3: {RV1681AreaNegativeStd_Q3}')
print(f'Area negative mean RV1681 Q4: {RV1681AreaNegativeMean_Q4}')
print(f'Std negative RV1681 Q4: {RV1681AreaNegativeStd_Q4}')
# %%
CFP10_TotalAreaPositiveMean = CFP10_Total[CFP10_Total['diagnostic'] == 'P']['area'].mean(
)
CFP10_TotalAreaPositiveStd = CFP10_Total[CFP10_Total['diagnostic'] == 'P']['area'].std(
)
CFP10_TotalAreaNegativeMean = CFP10_Total[CFP10_Total['diagnostic'] == 'N']['area'].mean(
)
CFP10_TotalAreaNegativeStd = CFP10_Total[CFP10_Total['diagnostic'] == 'N']['area'].std(
)
print(f'Area positive mean CFP10_Total: {CFP10_TotalAreaPositiveMean}')
print(f'Std CFP10_Total: {CFP10_TotalAreaPositiveStd}')
print(f'Area Negative mean CFP10_Total: {CFP10_TotalAreaNegativeMean}')
print(f'Std CFP10_Total: {CFP10_TotalAreaNegativeStd}')
f, (ax_box1, ax_box2, ax_hist) = plt.subplots(
    3, sharex=True, gridspec_kw={"height_ratios": (.15, .15, .70)})
sns.boxplot(CFP10_Total[CFP10_Total['diagnostic'] == 'P']
            ['area'], ax=ax_box1, color="plum")
sns.boxplot(CFP10_Total[CFP10_Total['diagnostic'] == 'N']
            ['area'], ax=ax_box2, color="g")
plt.title('CFP10_Total')
sns.distplot(CFP10_Total[CFP10_Total['diagnostic'] == 'P']['area'],
             label='P', bins=100, kde=False, ax=ax_hist, color="plum")
sns.distplot(CFP10_Total[CFP10_Total['diagnostic'] == 'N']['area'],
             label='N', bins=100, kde=False, ax=ax_hist, color="g")
ax_box1.set(xlabel='')
ax_box2.set(xlabel='')
# %%
ESAT6_TotalAreaPositiveMean = ESAT6_Total[ESAT6_Total['diagnostic'] == 'P']['area'].mean(
)
ESAT6_TotalAreaPositiveStd = ESAT6_Total[ESAT6_Total['diagnostic'] == 'P']['area'].std(
)
ESAT6_TotalAreaNegativeMean = ESAT6_Total[ESAT6_Total['diagnostic'] == 'N']['area'].mean(
)
ESAT6_TotalAreaNegativeStd = ESAT6_Total[ESAT6_Total['diagnostic'] == 'N']['area'].std(
)
print(f'Area positive mean ESAT6_Total: {ESAT6_TotalAreaPositiveMean}')
print(f'Std ESAT6_Total: {ESAT6_TotalAreaPositiveStd}')
print(f'Area Negative mean ESAT6_Total: {ESAT6_TotalAreaNegativeMean}')
print(f'Std ESAT6_Total: {ESAT6_TotalAreaNegativeStd}')
f, (ax_box1, ax_box2, ax_hist) = plt.subplots(
    3, sharex=True, gridspec_kw={"height_ratios": (.15, .15, .70)})
sns.boxplot(ESAT6_Total[ESAT6_Total['diagnostic'] == 'P']
            ['area'], ax=ax_box1, color="plum")
sns.boxplot(ESAT6_Total[ESAT6_Total['diagnostic'] == 'N']
            ['area'], ax=ax_box2, color="g")
plt.title('ESAT6_Total')
sns.distplot(ESAT6_Total[ESAT6_Total['diagnostic'] == 'P']['area'],
             label='P', bins=100, kde=False, ax=ax_hist, color="plum")
sns.distplot(ESAT6_Total[ESAT6_Total['diagnostic'] == 'N']['area'],
             label='N', bins=100, kde=False, ax=ax_hist, color="g")
ax_box1.set(xlabel='')
ax_box2.set(xlabel='')
# %%
RV1681_TotalAreaPositiveMean = RV1681_Total[RV1681_Total['diagnostic'] == 'P']['area'].mean(
)
RV1681_TotalAreaPositiveStd = RV1681_Total[RV1681_Total['diagnostic'] == 'P']['area'].std(
)
RV1681_TotalAreaNegativeMean = RV1681_Total[RV1681_Total['diagnostic'] == 'N']['area'].mean(
)
RV1681_TotalAreaNegativeStd = RV1681_Total[RV1681_Total['diagnostic'] == 'N']['area'].std(
)
print(f'Area positive mean RV: {RV1681_TotalAreaPositiveMean}')
print(f'Std RV: {RV1681_TotalAreaPositiveStd}')
print(f'Area Negative mean RV: {RV1681_TotalAreaNegativeMean}')
print(f'Std RV: {RV1681_TotalAreaNegativeStd}')
f, (ax_box1, ax_box2, ax_hist) = plt.subplots(
    3, sharex=True, gridspec_kw={"height_ratios": (.15, .15, .70)})
sns.boxplot(RV1681_Total[RV1681_Total['diagnostic'] == 'P']
            ['area'], ax=ax_box1, color="plum")
sns.boxplot(RV1681_Total[RV1681_Total['diagnostic']
                         == 'N']['area'], ax=ax_box2, color="g")
plt.title('RV1681_Total')
sns.distplot(RV1681_Total[RV1681_Total['diagnostic'] == 'P']['area'],
             label='P', bins=100, kde=False, ax=ax_hist, color="plum")
sns.distplot(RV1681_Total[RV1681_Total['diagnostic'] == 'N']['area'],
             label='N', bins=100, kde=False, ax=ax_hist, color="g")
ax_box1.set(xlabel='')
ax_box2.set(xlabel='')
# %%
# Anything above these thresholds will be positive, otherwise, negative
results = {
    'ESAT6': {
        'Negatives': {
            'Means': {
                'Total': ESAT6_TotalAreaNegativeMean,
                'ESAT6_Q1': ESAT6AreaNegativeMean_Q1,
                'ESAT6_Q2': ESAT6AreaNegativeMean_Q2,
                'ESAT6_Q3': ESAT6AreaNegativeMean_Q3,
                'ESAT6_Q4': ESAT6AreaNegativeMean_Q4
            },
            'Stds': {
                'Total': ESAT6_TotalAreaNegativeStd,
                'ESAT6_Q1': ESAT6AreaNegativeStd_Q1,
                'ESAT6_Q2': ESAT6AreaNegativeStd_Q2,
                'ESAT6_Q3': ESAT6AreaNegativeStd_Q3,
                'ESAT6_Q4': ESAT6AreaNegativeStd_Q4
            }
        },
        'Positives': {
            'Means': {
                'Total': ESAT6_TotalAreaPositiveMean,
                'ESAT6_Q1': ESAT6AreaPositiveMean_Q1,
                'ESAT6_Q2': ESAT6AreaPositiveMean_Q2,
                'ESAT6_Q3': ESAT6AreaPositiveMean_Q3,
                'ESAT6_Q4': ESAT6AreaPositiveMean_Q4
            },
            'Stds': {
                'Total': ESAT6_TotalAreaPositiveStd,
                'ESAT6_Q1': ESAT6AreaPositiveStd_Q1,
                'ESAT6_Q2': ESAT6AreaPositiveStd_Q2,
                'ESAT6_Q3': ESAT6AreaPositiveStd_Q3,
                'ESAT6_Q4': ESAT6AreaPositiveStd_Q4
            }
        }
    },
    'CFP10': {
        'Negatives': {
            'Means': {
                'Total': CFP10_TotalAreaNegativeMean,
                'CFP10_Q1': CFP10AreaNegativeMean_Q1,
                'CFP10_Q2': CFP10AreaNegativeMean_Q2,
                'CFP10_Q3': CFP10AreaNegativeMean_Q3,
                'CFP10_Q4': CFP10AreaNegativeMean_Q4
            },
            'Stds': {
                'Total': CFP10_TotalAreaNegativeStd,
                'CFP10_Q1': CFP10AreaNegativeStd_Q1,
                'CFP10_Q2': CFP10AreaNegativeStd_Q2,
                'CFP10_Q3': CFP10AreaNegativeStd_Q3,
                'CFP10_Q4': CFP10AreaNegativeStd_Q4
            }
        },
        'Positives': {
            'Means': {
                'Total': CFP10_TotalAreaPositiveMean,
                'CFP10_Q1': CFP10AreaPositiveMean_Q1,
                'CFP10_Q2': CFP10AreaPositiveMean_Q2,
                'CFP10_Q3': CFP10AreaPositiveMean_Q3,
                'CFP10_Q4': CFP10AreaPositiveMean_Q4
            },
            'Stds': {
                'Total': CFP10_TotalAreaPositiveStd,
                'CFP10_Q1': CFP10AreaPositiveStd_Q1,
                'CFP10_Q2': CFP10AreaPositiveStd_Q2,
                'CFP10_Q3': CFP10AreaPositiveStd_Q3,
                'CFP10_Q4': CFP10AreaPositiveStd_Q4
            }
        }
    },
    'RV1681': {
        'Negatives': {
            'Means': {
                'Total': RV1681_TotalAreaNegativeMean,
                'RV1681_Q1': RV1681AreaNegativeMean_Q1,
                'RV1681_Q2': RV1681AreaNegativeMean_Q2,
                'RV1681_Q3': RV1681AreaNegativeMean_Q3,
                'RV1681_Q4': RV1681AreaNegativeMean_Q4
            },
            'Stds': {
                'Total': RV1681_TotalAreaNegativeStd,
                'RV1681_Q1': RV1681AreaNegativeStd_Q1,
                'RV1681_Q2': RV1681AreaNegativeStd_Q2,
                'RV1681_Q3': RV1681AreaNegativeStd_Q3,
                'RV1681_Q4': RV1681AreaNegativeStd_Q4
            }
        },
        'Positives': {
            'Means': {
                'Total': RV1681_TotalAreaPositiveMean,
                'RV1681_Q1': RV1681AreaPositiveMean_Q1,
                'RV1681_Q2': RV1681AreaPositiveMean_Q2,
                'RV1681_Q3': RV1681AreaPositiveMean_Q3,
                'RV1681_Q4': RV1681AreaPositiveMean_Q4
            },
            'Stds': {
                'Total': RV1681_TotalAreaPositiveStd,
                'RV1681_Q1': RV1681AreaPositiveStd_Q1,
                'RV1681_Q2': RV1681AreaPositiveStd_Q2,
                'RV1681_Q3': RV1681AreaPositiveStd_Q3,
                'RV1681_Q4': RV1681AreaPositiveStd_Q4
            }
        }
    },
}
# %%
outputFile = 'thresholds.json'
file = open(outputFile, 'w')
file.write(json.dumps(results, indent=2))
file.close()
# %%
RV1681_Total2 = RV1681_Total[(RV1681_Total['diagnostic'] == 'P') & (
    RV1681_Total['area'] > 70)]
