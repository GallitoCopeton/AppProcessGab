def tbDiagnostic(xmarker1, xmarker2, xmarker3):  # ESAT6, CFP10, RV1681
    if (xmarker1 == 'P' or xmarker2 == 'P' or xmarker3 == 'P'):
        return 'P'
    elif(xmarker1 == 'N' and xmarker2 == 'N' and xmarker3 == 'N'):
        return 'N'
    else:
        return 'I'

# DEPRECATED


def blobAnalysis(imageBin, keypoints):
    if(not isEmpty(imageBin)):
        if(len(keypoints) > 2):
            return 'P'
        if(len(keypoints) < 2):
            return 'N'
        else:
            return 'I'
    else:
        return 'E'

# DEPRECATED


def areaAnalysis(imageBin, keypoints, area, nmarker):
    # ths = [26,30,36,26,35,22]
    # ths = [45,48,47,36,35,30]
    ths = [45, 48, 45, 37, 35, 35]
    if(not isEmpty(imageBin)):
        # Un sólo blob siempre es negativo
        if (area < ths[nmarker]) or (len(keypoints) < 2):
            return 'N'
        else:
            return 'P'
    else:
        return 'E'


# DEPRECATED


def areaAnalysis2(imageBin, blobs, area, markerName):
    thresholds = {
        'ESAT6': 45,
        'CFP10': 48,
        'RV1681': 45,
        'P24': 37,
        'GP120': 35,
        'Control': 35,
    }
    # Un sólo blob siempre es negativo
    if len(blobs) <= 1:
        return 'N'
    else:
        if area < thresholds[markerName]:
            return 'N'
        else:
            return 'P'
