# %%
import base64
import os

import cv2
import pandas as pd

import qrQuery
from IF2.Crop import croppingProcess as cP
from IF2.ReadImage import readImage as rI
from IF2.Shows import showProcesses as sP
from IF2.Processing import colorTransformations as cT

# %%
zeptoURI = 'mongodb://validationUser:85d4s32D2%23diA@idenmon.zapto.org:888/findValidation?authSource=findValidation'
zeptoDbName = 'findValidation'
zeptoCollectionRegistersName = 'registerstotals'
zeptoCollectionImagesName = 'imagestotals'
zeptoDataCollection = qrQuery.getCollection(
    zeptoURI, zeptoDbName, zeptoCollectionRegistersName)
zeptoImagesCollection = qrQuery.getCollection(
    zeptoURI, zeptoDbName, zeptoCollectionImagesName)
zeptoDatabase = qrQuery.getDatabase(zeptoURI, zeptoDbName)
newMarkerCollectionName = 'markerTotals'
# zeptoDatabase.create_collection(newMarkerCollectionName)
newMarkerCollection = qrQuery.getCollection(
    zeptoURI, zeptoDbName, newMarkerCollectionName)
# %%
elisaExcel = pd.read_excel(
    'Relación de resultados de dispositivos con ELISA de sueros de Zepto (1).xlsx')
data = elisaExcel.iloc[4:, 4:]
columns = ['iter1', 'iter2', 'iter3', 'suero', 'repeatabilityE6',
           'repeatabilityCF', 'repeatabilityRV', 'diagE6', 'diagCF', 'diagRV']
data.columns = columns
noNan = data.dropna()
# %%
insertedQRFile = 'insertedQrs.txt'
if not os.path.isfile(insertedQRFile):
    with open(insertedQRFile, 'w') as file:
        file.write('')
with open(insertedQRFile, 'r') as file:
    insertedQrs = file.read().split('\n')[:-1]
markerNames = ['ESAT6', 'CFP10', 'RV1681']
for i, row in noNan.iterrows():
    print(f'Iteraciones del suero número {row.suero}')
    qrIter1 = str(int(row.iter1))
    data1 = zeptoDataCollection.find_one({'qrCode': qrIter1})
    if data1 is not None and data1['control'] == 'Valid':
        image1 = rI.customQuery(zeptoImagesCollection, {'$and': [
            {'fileName': qrIter1},
            {'fileName': {'$nin': insertedQrs}}
        ]
        })
        if len(image1) > 0:
            for image in image1:
                count1 = image['count']
                image = cT.BGR2RGB(image['file'])
                diagE6 = 'P' if row.diagE6 == '+' else 'N'
                diagCF = 'P' if row.diagCF == '+' else 'N'
                diagRV = 'P' if row.diagRV == '+' else 'N'
                diagnostics = [diagE6, diagCF, diagRV]
                try:
                    testSite1 = cP.getTestArea(image)
                except Exception as e:
                    print(e)
                    pass
                try:
                    markers1 = cP.getMarkers(testSite1)
                except Exception as e:
                    print(e)
                    pass
                print(f'Imagen 1 del qr {qrIter1} count {count1}')
                sP.showImage(
                    testSite1, title=f'E6: {diagE6} CF: {diagCF} RV: {diagRV}', figSize=(6, 6))
                for marker, name, diagnostic in zip(markers1, markerNames, diagnostics):
                    #                sP.showImage(marker, title=f'{name}', figSize=(3,3))
                    markerToInsert = {}
                    markerToInsert['marker'] = name
                    markerToInsert['count'] = count1
                    markerToInsert['QR'] = qrIter1
                    retval, buffer = cv2.imencode('.png', marker)
                    text = base64.b64encode(buffer)
                    markerToInsert['image'] = text
                    markerToInsert['diagnostic'] = diagnostic
                    sP.showImage(marker, title=name)
                    newMarkerCollection.insert_one(markerToInsert)
    with open(insertedQRFile, 'a') as file:
        file.write(qrIter1 + '\n')
    qrIter2 = str(int(row.iter2))
    data2 = zeptoDataCollection.find_one({'qrCode': qrIter2})
    if data2 is not None and data2['control'] == 'Valid':
        image2 = rI.customQuery(zeptoImagesCollection, {'$and': [
            {'fileName': qrIter2},
            {'fileName': {'$nin': insertedQrs}}
        ]
        })
        if len(image2) > 0:
            for image in image2:
                count2 = image['count']
                image = cT.BGR2RGB(image['file'])
                diagE6 = 'P' if row.diagE6 == '+' else 'N'
                diagCF = 'P' if row.diagCF == '+' else 'N'
                diagRV = 'P' if row.diagRV == '+' else 'N'
                diagnostics = [diagE6, diagCF, diagRV]
                try:
                    testSite2 = cP.getTestArea(image)
                except Exception as e:
                    print(e)
                    pass
                try:
                    markers2 = cP.getMarkers(testSite2)
                except Exception as e:
                    print(e)
                    pass
                print(f'Imagen 2 del qr {qrIter2} count {count2}')
                sP.showImage(
                    testSite2, title=f'E6: {diagE6} CF: {diagCF} RV: {diagRV}', figSize=(6, 6))
                for marker, name, diagnostic in zip(markers2, markerNames, diagnostics):
                    #                sP.showImage(marker, title=f'{name}', figSize=(3,3))
                    markerToInsert = {}
                    markerToInsert['marker'] = name
                    markerToInsert['count'] = count2
                    markerToInsert['QR'] = qrIter2
                    retval, buffer = cv2.imencode('.png', marker)
                    text = base64.b64encode(buffer)
                    markerToInsert['image'] = text
                    markerToInsert['diagnostic'] = diagnostic
                    sP.showImage(marker, title=name)
                    newMarkerCollection.insert_one(markerToInsert)
    with open(insertedQRFile, 'a') as file:
        file.write(qrIter2 + '\n')
    qrIter3 = str(int(row.iter3))
    data3 = zeptoDataCollection.find_one({'qrCode': qrIter3})
    if data3 is not None and data3['control'] == 'Valid':
        image3 = rI.customQuery(zeptoImagesCollection, {'$and': [
            {'fileName': qrIter3},
            {'fileName': {'$nin': insertedQrs}}
        ]
        })
        if len(image3) > 0:
            for image in image3:
                count3 = image['count']
                image = cT.BGR2RGB(image['file'])
                diagE6 = 'P' if row.diagE6 == '+' else 'N'
                diagCF = 'P' if row.diagCF == '+' else 'N'
                diagRV = 'P' if row.diagRV == '+' else 'N'
                diagnostics = [diagE6, diagCF, diagRV]
                try:
                    testSite3 = cP.getTestArea(image)
                except Exception as e:
                    print(e)
                    pass
                try:
                    markers3 = cP.getMarkers(testSite3)
                except Exception as e:
                    print(e)
                    pass
                print(f'Imagen 3 del qr {qrIter3} count {count3}')
                sP.showImage(
                    testSite3, title=f'E6: {diagE6} CF: {diagCF} RV: {diagRV}', figSize=(6, 6))
                for marker, name, diagnostic in zip(markers3, markerNames, diagnostics):
                    #                sP.showImage(marker, title=f'{name}', figSize=(3,3))
                    markerToInsert = {}
                    markerToInsert['marker'] = name
                    markerToInsert['count'] = count3
                    markerToInsert['QR'] = qrIter3
                    retval, buffer = cv2.imencode('.png', marker)
                    text = base64.b64encode(buffer)
                    markerToInsert['image'] = text
                    markerToInsert['diagnostic'] = diagnostic
                    sP.showImage(marker, title=name)
                    newMarkerCollection.insert_one(markerToInsert)
    with open(insertedQRFile, 'a') as file:
        file.write(qrIter3 + '\n')

