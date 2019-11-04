import cv2

def blobDetect(image):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.thresholdStep = 10
    params.minThreshold = 50
    params.maxThreshold = 220

    # Change repeatability
    params.minRepeatability = 2

    # Distance between blobs
    params.minDistBetweenBlobs = 10

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 5000

    # Filter by color
    params.filterByColor = True
    params.blobColor = 0

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 8.0000001192092896e-001
    params.maxCircularity = 3.4028234663852886e+038

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.05
    params.maxConvexity = 1.5

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 3.4028234663852886e+038

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Keypoints
    return detector.detect(image)


def drawBlobs(image, keypoints):
    return cv2.drawKeypoints(image, keypoints, np.array([]),
                             (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def areaEstimation(blobs):
    return np.array([blob.size for blob in blobs]).sum()


def getBlobInfoTB(markersNot):
    sites = ['ESAT6', 'CF', 'RV', 'Control']
    results = [
        blobAnalysis(markerNot, blobDetect(markerNot))
        for markerNot in markersNot
    ]
    return pd.DataFrame(results, index=sites, columns=['Pixels'])
