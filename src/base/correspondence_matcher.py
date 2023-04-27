import cv2
import numpy as np

class Correspondence_Matcher():
    _detector_ = None
    _extractor_ = None
    _matcher_ = None
    _ratio_ = None
    _training_img_ = None
    _img_matching_ = None

    def __init__(self):
        self._ratio_ = 0.8
        self._training_img_ = None
        self._img_matching_ = None
    
        self._detector_ = cv2.ORB_create()
        self._extractor_ = cv2.ORB_create()

        self._matcher_ = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)

    def setFeatureDetector(self, detect):
        self._detector_ = detect

    def setDescriptorExtractor(self, desc):
        self._extractor_ = desc

    def setDescriptorMatcher(self, match):
        self._matcher_ = match

    def computeKeyPoints(self, image):
        keypoints = self._detector_.detect(image, None)
        return keypoints

    def computeDescriptors(self, image, keypoints):
        descriptors = None
        keypoints, descriptors = self._extractor_.compute(image, keypoints)
        return descriptors

    def getImageMatching(self):
        return self._img_matching_

    def setRatio(self, rat : float):
        self._ratio_ = rat

    def setTrainingImage(self, img : cv2.Mat):
        self._training_img_ = img

    def ratioTest(self, matches):
        removed = 0
        matches = list(matches)
        for matchIterator in matches:
            if (len(matchIterator) > 1):
                if ((matchIterator)[0].distance / (matchIterator)[1].distance > self._ratio_):
                    matches.remove(matchIterator)
                    removed += 1
            else:
                matches.remove(matchIterator)
                removed += 1
        return removed

    def robustMatch(self, frame, descriptors_model, keypoints_model):
        good_matches = []
        keypoints_frame = []
        keypoints_frame = self.computeKeyPoints(frame)

        descriptors_frame = None
        descriptors_frame = self.computeDescriptors(frame, keypoints_frame)

        matches = None
        matches = self._matcher_.knnMatch(descriptors_frame, descriptors_model, 2)

        self.ratioTest(matches)

        for matchIterator in matches:
            if (matchIterator):
                good_matches.append((matchIterator)[0])

        if (self._training_img_ is not None and keypoints_model is not None):
            cv2.drawMatches(frame, keypoints_frame, self._training_img_, keypoints_model, good_matches, self._img_matching_)

        return good_matches, keypoints_frame
