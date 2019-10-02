import numpy as np
import cv2
import operator
import sys 

#Scripts para leer y procesar imagen
sys.path.insert(0, '../Golden Master (AS IS)')
import readImage
import preProcessing as pP
import sorts as srt
import perspective as pPe
import indAnalysis as inA


##Find XM Group
def findXMGroup(img):
    if(isinstance(img[0], str) or isinstance(img[0], list)): #Image not found
        return img[0]
    #imgBinary = pP.contourBinarization(img, 3, 7, 85, 2, inverse = True, mean = False) #Gaussian Adaptative
    #externalContours = pP.findExternalContours(imgBinary)
    externalContours = img[1]
    #print(len(externalContours))
    if(len(externalContours) == 2):
        maxCnt = max(externalContours, key = cv2.contourArea)
        return pPe.perspectiveTransform(img[0], srt.sortPoints(maxCnt), 5)
    else:
        imgBinary = pP.contourBinarization(img[0], 3, 7, 85, 2, inverse = True, mean = False) #Gaussian Adaptative
        externalContours = pP.findExternalContours(imgBinary) 
        if(len(externalContours) == 2):
            maxCnt = max(externalContours, key = cv2.contourArea)
            return pPe.perspectiveTransform(img[0], srt.sortPoints(maxCnt), 5)
        else:
            return "Complete tests image did not found, tests: " + str(len(externalContours))

##Find Biggest Square
def biggestSquare(img):
    if(isinstance(img, str)): #Image not found
        return img
    img = pP.resizeImg(img, 728)
    #Gaussian Adaptative
    imgBinary = pP.contourBinarization(img, 3, 7, 85, 2, inverse = True, mean = False) 
    externalSquare = pP.findTreeContours(imgBinary)
    externalOrdSquare = srt.sortPointsContours(externalSquare)
    i = 0
    perspective = []
    external = []
    perspectiveBGR = []
    while(len(external) != 2 and i < (len(externalOrdSquare))):
        perspectiveCopy = perspective.copy()
        perspectiveBGRCopy = perspectiveBGR.copy()

        perspective = pPe.perspectiveTransform(imgBinary, externalOrdSquare[i], -5, True)
        perspectiveBGR = pPe.perspectiveTransform(img, externalOrdSquare[i], -5)
        external = pP.findExternalContours(perspective)

        if (len(external)==0 and i > 0):
            perspective = perspectiveCopy
            perspectiveBGR = perspectiveBGRCopy
        #print(str(i) + ", " + str(len(external)))
        i += 1
    return (perspectiveBGR, external)

### Find Individual X-Markers Boxes (a.k.a. sitios de prueba)
def findXMBox(img):
    if(isinstance(img, str) or isinstance(img, list)): #If string received
        return img
    ## Histogram Equalization
    img = pP.equalizeHistogram(img)

    height, width = img.shape[:2]
    areaInd = (height*width/8) - 10
    #Binarization is required again because lost of details on first binarization
    contoursInd = pP.findTreeContours(pP.contourBinarization(img, 3, 7, 85, 2, mean = False),areaInd)
    if(len(contoursInd) == 7 ): 
        contoursInd = contoursInd[1:]
    
    #print('Number of Xmarkers: ' + str(len(contoursInd)))

    if(len(contoursInd) == 6):
        srt.sortTests(contoursInd)
        listXMarkers = []
        for i,c in enumerate(contoursInd):
            test = pPe.getIndTest(img,c)
            listXMarkers.append(test)
    else:
        return 'Error finding individual tests!'
    return listXMarkers

#Get individual X-Markers Boxes in binary format
def getXMarkersBinary(listXMarkers):
    if(isinstance(listXMarkers, str)): #If string received
        return listXMarkers
    listXMarkersOut = []
    #Tests individuals, resized binary
    for i, test in enumerate(listXMarkers):
        testBin = pP.contourBinarization(test, 3, 3, 45, 4, Gs = 0, inverse = False, mean = True)
        listXMarkersOut.append(testBin)
    return listXMarkersOut

#Get individual X-Markers Boxes with blobs
def getXMarkersBlobs(listXMarkers):
    if(isinstance(listXMarkers, str)): #If string received
        return listXMarkers
    mask = inA.readMask()
    
    listXMarkersOut = []
    #Tests individuals, resized binary
    for i, test in enumerate(listXMarkers):
        testBinMask = inA.andOperation(test, mask)
        testBinMaskEroDil = inA.erosionDilation(testBinMask, 3)
        testBinMaskEroDilNot = cv2.bitwise_not(testBinMaskEroDil)
        listXMarkersOut.append(testBinMaskEroDilNot)
    return listXMarkersOut

## X-Marker Analisis (a.k.a. sitios de prueba)
def analizeXMBlobs(listXMarkers):
    listXMarkers = inA.resizeAll(listXMarkers)
    if len(listXMarkers) != 6:
        return 'Problem to find all tests'
    listBinaryXMarkers = getXMarkersBinary(listXMarkers)
    listXMarkersBlobs = getXMarkersBlobs(listBinaryXMarkers)
    control = listXMarkers[5]

    if(isinstance(listXMarkersBlobs, str)): #If string received
        return listXMarkersBlobs
        
    statsH, statsS, statsV = inA.controlStats(control) #Hue and Saturation control stats

    dict_test = dict()
    markerNames = ["ESAT6","CFP10","RV1681","P24","H2", "Control"]
    for i,img in enumerate(listXMarkersBlobs):
        blobs = inA.blobDetect(img)
        dict_test[markerNames[i] + "_blobs"] = len(blobs)
        dict_test[markerNames[i] + "_area"] = inA.areaEstimation(blobs)
        dict_test[markerNames[i] + "_area_color"] = inA.colorSegmentation(listXMarkers[i], statsH, statsS)[2]
        #Resultado algoritmo conteo blobs
        dict_test[markerNames[i] + "_result_blobs"] = inA.blobAnalysis(listBinaryXMarkers[i], blobs)
        '''
        #Resultado algoritmo areas de blobs
        dict_test[markerNames[i] + "_result_area"] = inA.areaAnalysis(listBinaryXMarkers[i], 
            dict_test[markerNames[i] + "_area"], i)
        '''
        #Resultado algoritmo areas de blobs (1 Solo Blob siempre es negativo)
        dict_test[markerNames[i] + "_result_area"] = inA.areaAnalysis(listBinaryXMarkers[i], blobs,
            dict_test[markerNames[i] + "_area"], i)
    dict_test['tb_result_area'] = inA.tbDiagnostic(dict_test[markerNames[0]+ "_result_area"], 
        dict_test[markerNames[1]+ "_result_area"], dict_test[markerNames[2]+ "_result_area"])
    return dict_test

def cropXMarkers(img):
    outBS = biggestSquare(img)
    outXMG = findXMGroup(outBS)
    outXMB = findXMBox(outXMG)
    return outXMB

def completePipeline(qr):
    ### Read Image
    outputRI = readImage.readImage(qr)

    if(isinstance(outputRI, str)):
        return(outputRI)

    ## Return 1 if it is blurry
    blurry = pP.isBlurry(outputRI)

    ## Find Biggest Square
    ##  Find Counturs Biggest Square
    ##  Sort points of every contour
    ##  Perspective Transform for Biggest Square
    inputBS = outputRI
    outputBS = biggestSquare(inputBS)

    ## Find X-Markers Group (a.k.a. Test Square)
    ##   Set perspective X-Markers Group
    ##   Adjust Offset X-Markers Group
    ##   Binarize X-Markers Group 
    ##   Find Contours using Binarized X-Markers Group
    ##   Keep copy of RGB image for later use
    inputXG = outputBS
    outputXG = findXMGroup(inputXG)

    ### Find Individual X-Markers Boxes (a.k.a. sitios de prueba)
    inputXB = outputXG
    outputXB = findXMBox(inputXB)

    ## X-Marker Analisis (a.k.a. sitios de prueba)
    ##   Apply Mask
    ##   Erosion Dilation
    ##   Blob Detection
    ##   Assses Blob Areas and Assign Results
    inputXA = outputXB
    inputXA = analizeXMBlobs(inputXA)

    ### Image Quality Analysis
    if (not isinstance(inputXA,str)):
        inputXA['blurry'] = blurry

    return inputXA

def multiPipeline(qr):
    auxList = []
    listResults = []

    ### Read Images
    images = readImage.readImage(qr, array = True)
    
    if (not isinstance(images,list)):
        auxList.append(images)
        images = auxList

    for i,outputRI in enumerate(images):

        if(isinstance(outputRI, str)):
            continue

        ## Return 1 if it is blurry
        blurry = pP.isBlurry(outputRI)

        ## Find Biggest Square
        ##  Find Counturs Biggest Square
        ##  Sort points of every contour
        ##  Perspective Transform for Biggest Square
        inputBS = outputRI
        outputBS = biggestSquare(inputBS)

        ## Find X-Markers Group (a.k.a. Test Square)
        ##   Set perspective X-Markers Group
        ##   Adjust Offset X-Markers Group
        ##   Binarize X-Markers Group 
        ##   Find Contours using Binarized X-Markers Group
        ##   Keep copy of RGB image for later use
        inputXG = outputBS
        outputXG = findXMGroup(inputXG)

        ### Find Individual X-Markers Boxes (a.k.a. sitios de prueba)
        inputXB = outputXG
        outputXB = findXMBox(inputXB)

        ## X-Marker Analisis (a.k.a. sitios de prueba)
        ##   Apply Mask
        ##   Erosion Dilation
        ##   Blob Detection
        ##   Assses Blob Areas and Assign Results
        inputXA = outputXB
        inputXA = analizeXMBlobs(inputXA)

        ### Image Quality Analysis
        if (not isinstance(inputXA,str)):
            inputXA['blurry'] = blurry
            inputXA['count'] = i
            listResults.append(inputXA)

    return listResults