import pandas as pd

import qrQuery

# %%
# Real: base con diagnósticos reales Zepto: base de pruebas de Zeptometrix Clean: base con marcadores seleccionados
zeptoURI = 'mongodb://validationUser:85d4s32D2%23diA@idenmon.zapto.org:888/findValidation?authSource=findValidation'
zeptoDbName = 'findValidation'
zeptoCollectionRegistersName = 'registerstotals'
zeptoDataCollection = qrQuery.getCollection(
    zeptoURI, zeptoDbName, zeptoCollectionRegistersName)
# %%
limit = 0
data = zeptoDataCollection.find({'marker': {'$size': 4}}).limit(limit)
completeDf = []
for register in data:
    registerDict = {}
    registerDict['QR'] = register['qrCode']
    registerDict['Fecha'] = register['date']
    registerDict['Count'] = register['count']
    for marker in register['marker']:
        registerDict[marker['name']] = marker['result']
    completeDf.append(pd.DataFrame.from_records([registerDict]))
completeDf = pd.concat(completeDf, sort=False)
completeDf.to_excel('baseDeDatosZepto.xlsx', index=False)
