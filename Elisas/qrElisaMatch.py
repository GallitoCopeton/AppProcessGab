# %%
import base64

import pandas as pd
import cv2

import qrQuery
from AppProcess.CroppingProcess import croppingProcess as cP
from ReadImages import readImage as rI
from ShowProcess import showProcesses as sP
#%%
zeptoURI = 'mongodb://validationUser:85d4s32D2%23diA@idenmon.zapto.org:888/findValidation?authSource=findValidation'
zeptoDbName = 'findValidation'
zeptoCollectionRegistersName = 'registerstotals'
zeptoCollectionImagesName = 'imagestotals'
zeptoImagesCollection = qrQuery.getCollection(
    zeptoURI, zeptoDbName, zeptoCollectionRegistersName)
zeptoImagesCollection = qrQuery.getCollection(
    zeptoURI, zeptoDbName, zeptoCollectionImagesName)
zeptoDatabase = qrQuery.getDatabase(zeptoURI, zeptoDbName)
newMarkerCollectionName = 'markerTotals'
#zeptoDatabase.create_collection(newMarkerCollectionName)
newMarkerCollection = qrQuery.getCollection(zeptoURI, zeptoDbName, newMarkerCollectionName)
#%%
elisaExcel = pd.read_excel('Relación de resultados de dispositivos con ELISA de sueros de Zepto (1).xlsx')
data = elisaExcel.iloc[4:, 4:]
columns = ['iter1', 'iter2', 'iter3', 'suero', 'repeatabilityE6', 'repeatabilityCF', 'repeatabilityRV', 'diagE6', 'diagCF', 'diagRV']
data.columns = columns
noNan = data.dropna()
#%%
insertedQRFile = 'insertedQrs.txt'
with open(insertedQRFile, 'r') as file:
    insertedQrs = file.read().split('\n')[:-1]
markerNames = ['ESAT6', 'CFP10', 'RV1681']
for i, row in noNan.iterrows():
    qrIter1 = str(int(row.iter1))
    image1 = rI.customQuery(zeptoImagesCollection, {'$and': [
            {'fileName': qrIter1},
            {'fileName': {'$nin': insertedQrs}}
            ]
    })
    if len(image1) > 0:
        count1 = image1[0]['count']
        image1 = image1[0]['file']
        diagE6 = 'P' if row.diagE6 == '+' else 'N'
        diagCF = 'P' if row.diagCF == '+' else 'N'
        diagRV = 'P' if row.diagRV == '+' else 'N'
        diagnostics = [diagE6, diagCF, diagRV]
        try:
            testSite1 = cP.getTestArea(image1)
        except Exception as e:
            pass
        try:
            markers1 = cP.getMarkers(testSite1)
        except Exception as e:
            pass
        for marker, name, diagnostic in zip(markers1, markerNames, diagnostics):
            markerToInsert = {}
            markerToInsert['marker'] = name
            markerToInsert['count'] = count1
            markerToInsert['QR'] = qrIter1
            retval, buffer = cv2.imencode('.png', marker)
            text = base64.b64encode(buffer)
            markerToInsert['image'] = text
            markerToInsert['diagnostic'] = diagnostic
            newMarkerCollection.insert_one(markerToInsert)
            with open(insertedQRFile, 'a') as file:
                file.write(qrIter1 + '\n')
        sP.showImage(testSite1, title=f'E6: {diagE6} CF: {diagCF} RV: {diagRV}', figSize=(6,6))
    qrIter2 = str(int(row.iter2))
    image2 = rI.customQuery(zeptoImagesCollection, {'$and': [
            {'fileName': qrIter2},
            {'fileName': {'$nin': insertedQrs}}
            ]
    })
    if len(image2) > 0:
        count2 = image2[0]['count']
        image2 = image2[0]['file']
        diagE6 = 'P' if row.diagE6 == '+' else 'N'
        diagCF = 'P' if row.diagCF == '+' else 'N'
        diagRV = 'P' if row.diagRV == '+' else 'N'
        diagnostics = [diagE6, diagCF, diagRV]
        try:
            testSite2 = cP.getTestArea(image2)
        except Exception as e:
            pass
        try:
            markers2 = cP.getMarkers(testSite2)
        except Exception as e:
            pass
        
        for marker, name, diagnostic in zip(markers2, markerNames, diagnostics):
            markerToInsert = {}
            markerToInsert['marker'] = name
            markerToInsert['count'] = count2
            markerToInsert['QR'] = qrIter2
            retval, buffer = cv2.imencode('.png', marker)
            text = base64.b64encode(buffer)
            markerToInsert['image'] = text
            markerToInsert['diagnostic'] = diagnostic
            newMarkerCollection.insert_one(markerToInsert)
            with open(insertedQRFile, 'a') as file:
                file.write(qrIter2 + '\n')
        sP.showImage(testSite2, title=f'E6: {diagE6} CF: {diagCF} RV: {diagRV}', figSize=(6,6))
    qrIter3 = str(int(row.iter3))
    image3 = rI.customQuery(zeptoImagesCollection, {'$and': [
            {'fileName': qrIter3},
            {'fileName': {'$nin': insertedQrs}}
            ]
    })
    if len(image3) > 0:
        count3 = image3[0]['count']
        image3 = image3[0]['file']
        diagE6 = 'P' if row.diagE6 == '+' else 'N'
        diagCF = 'P' if row.diagCF == '+' else 'N'
        diagRV = 'P' if row.diagRV == '+' else 'N'
        diagnostics = [diagE6, diagCF, diagRV]
        try:
            testSite3 = cP.getTestArea(image3)
        except Exception as e:
            pass
        try:
            markers3 = cP.getMarkers(testSite3)
        except Exception as e:
            pass
        for marker, name, diagnostic in zip(markers3, markerNames, diagnostics):
            markerToInsert = {}
            markerToInsert['marker'] = name
            markerToInsert['count'] = count3
            markerToInsert['QR'] = qrIter3
            retval, buffer = cv2.imencode('.png', marker)
            text = base64.b64encode(buffer)
            markerToInsert['image'] = text
            markerToInsert['diagnostic'] = diagnostic
            newMarkerCollection.insert_one(markerToInsert)
            with open(insertedQRFile, 'a') as file:
                file.write(qrIter3 + '\n')
        sP.showImage(testSite3, title=f'E6: {diagE6} CF: {diagCF} RV: {diagRV}', figSize=(6,6))
#    break
    
    
    