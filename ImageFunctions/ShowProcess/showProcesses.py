import numpy as np
from matplotlib import pyplot as plt

from ImageFunctions.AppProcess.CroppingProcess import croppingProcess as cP
from ImageFunctions.AppProcess.MarkerProcess import markerProcess as mP


def showOldProcess(originalPicture, figSizeTupple=(5, 5), show=False):
    try:
        testSiteEq = cP.getEqTestSite(originalPicture)
    except:
        print('An error with the test site cropping ocurred')
        return
    try:
        markersEq = cP.getMarkers(testSiteEq)
    except:
        print('An error with the marker cropping ocurred')
        return
    markersEq, markersBin, markersTrans, markersNot = mP.oldProcess(
        markersEq, extendedProcess=True)

    if show:
        iList = np.arange(0, len(markersEq), 1)
        fig = plt.figure(figsize=figSizeTupple, constrained_layout=True)
        gs = fig.add_gridspec(len(markersEq), 6)
        fig.set_facecolor((0.6, 0.6, 0.6))
        try:
            plotFullCard = fig.add_subplot(gs[:, :2])
        except:
            return
        plotFullCard.set_title('Original image')
        plotFullCard.set_axis_off()
        plotFullCard.imshow(testSiteEq)
        for (markerEq, markerBin, markerTrans, markerNot, i) in zip(markersEq, markersBin, markersTrans, markersNot, iList):
            # Plots
            plotMarkerEq = fig.add_subplot(gs[i, 1])
            plotMarkerBin = fig.add_subplot(gs[i, 2])
            plotMarkerTrans = fig.add_subplot(gs[i, 3])
            plotMarkerBlobs = fig.add_subplot(gs[i, 4])

            plotMarkerEq.set_title('Equalization')
            plotMarkerBin.set_title('Binarized')
            plotMarkerTrans.set_title('Morph')
            plotMarkerBlobs.set_title('Blobs')

            plotMarkerEq.set_axis_off()
            plotMarkerBin.set_axis_off()
            plotMarkerTrans.set_axis_off()
            plotMarkerBlobs.set_axis_off()

            plotMarkerEq.imshow(markerEq)
            plotMarkerBin.imshow(markerBin, 'gray')
            plotMarkerTrans.imshow(markerTrans, 'gray')
            plotMarkerBlobs.imshow(markerNot, 'gray')
        plt.show()
        plt.close(fig)
    return markersEq, markersBin, markersTrans, markersNot


def showClusterProcess(originalPicture, kColors, attempts, figSizeTupple=(5, 5), show=True):
    try:
        testSite = cP.getNonEqTestSite(originalPicture)
    except:
        print('An error with the test site cropping ocurred')
        return
    try:
        markers = cP.getMarkers(testSite)
    except:
        print('An error with the marker cropping ocurred')
        return
    markersRecon, markersBin, markersTrans, markersNot = mP.clusteringProcess(
        markers, kColors, attempts, extendedProcess=True)
    if show:
        iList = np.arange(0, len(markers), 1)
        fig = plt.figure(figsize=figSizeTupple, constrained_layout=True)
        gs = fig.add_gridspec(len(markers), 6)
        fig.set_facecolor((0.6, 0.6, 0.6))
        try:
            plotFullCard = fig.add_subplot(gs[:, :2])
        except:
            return
        plotFullCard.set_title('Original image')
        plotFullCard.set_axis_off()
        plotFullCard.imshow(testSite)
        for (markerRecon, markerBin, markerTrans, markerNot, i) in zip(markersRecon, markersBin, markersTrans, markersNot, iList):
            # Plots
            plotMarkerRecon = fig.add_subplot(gs[i, 2])
            plotMarkerBin = fig.add_subplot(gs[i, 3])
            plotMarkerTrans = fig.add_subplot(gs[i, 4])
            plotMarkerBlobs = fig.add_subplot(gs[i, 5])

            plotMarkerRecon.set_title('Reconstruction')
            plotMarkerBin.set_title('Masked Recon')
            plotMarkerTrans.set_title('Binary')
            plotMarkerBlobs.set_title('Blobs')

            plotMarkerRecon.set_axis_off()
            plotMarkerBin.set_axis_off()
            plotMarkerTrans.set_axis_off()
            plotMarkerBlobs.set_axis_off()

            plotMarkerRecon.imshow(markerRecon)
            plotMarkerBin.imshow(markerBin)
            plotMarkerTrans.imshow(markerTrans)
            plotMarkerBlobs.imshow(markerNot)

        plt.show()
        plt.close(fig)
    return markersRecon, markersBin, markersTrans, markersNot
