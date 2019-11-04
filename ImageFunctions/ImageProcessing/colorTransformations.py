import cv2


def BGR2gray(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_BGR2GRAY)


def BGR2HLS(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_BGR2HLS)


def BGR2HSV(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_BGR2HSV)


def BGR2LAB(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_BGR2LAB)


def BGR2RGB(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_BGR2RGB)


def BGR2YUV(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_BGR2YUV)


def RGB2BGR(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_RGB2BGR)


def RGB2gray(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_RGB2GRAY)


def RGB2HLS(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_RGB2HLS)


def RGB2HSV(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_RGB2HSV)


def RGB2LAB(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_RGB2LAB)


def RGB2YUV(threeChannelImage):
    return cv2.cvtColor(threeChannelImage, cv2.COLOR_RGB2YUV)
