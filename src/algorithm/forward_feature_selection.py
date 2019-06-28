import numpy as np

from src.mutual_information.feature_selection import FeatueSelector


class ForwardFeatureSelector(FeatueSelector):
    def __init__(self, miEstimator, classification=False, X=None, Y=None, nproc=1):
        super().__init__(miEstimator, classification, X, Y, nproc)

    def _findFirstFeature(self):
        """The first feature is the one that maximizes I(X_i; Y)
        """
        scores = np.zeros(self.original_features.shape[1])
        for i in range(self.original_features.shape[1]):
            scores[i] = self.miEstimator.estimateMI(
                self.original_features[:, i], self.responses)

        self.selectedInd = [np.argmax(scores)]
        self.selected = self.original_features[:, self.selectedInd]
        print("First feature score: {0}".format(np.max(scores)))
        return np.max(scores)

    def scoreFeatures(self):
        """
        Given the S selected features, each feature gets the score
        I(X_i, Y | S)
        """
        scores = np.zeros(self.current_features.shape[1])
        for i in range(self.current_features.shape[1]):
            scores[i] = self.miEstimator.estimateConditionalMI(
                self.current_features[:, i], self.responses, self.selected)
        featureAndScores = list(zip(range(len(scores)), scores))
        return sorted(featureAndScores, key=lambda x: x[1], reverse=True)

    def _addBestId(self, addedFeature):
        self.selectedInd.append(addedFeature)
        self.current_features = np.delete(
            self.original_features, self.selectedInd, axis=1)
        self.selected = self.original_features[:, self.selectedInd]
        for k, v in list(self.idMap.items())[:-1]:
            if k >= addedFeature:
                self.idMap[k] = self.idMap[k+1]
        self.idMap.pop(max(self.idMap))

    def _selectKFeatures(self, num_features):
        self._findFirstFeature()
        numSelected = 1
        while numSelected < num_features:
            scores = self.scoreFeatures()
            best = scores[0][0]
            self._addBestId(best)
            numSelected += 1
        return set(self.selectedInd)

    def _selectOnError(self, max_error):
        self.res = 0
        score = self._findFirstFeature()
        self.res += max(score, 0)
        error = self.computeError()
        while error < max_error:
            scores = self.scoreFeatures()
            best = scores[0][0]
            self._addBestId(best)
            self.res += max(scores[0][1], 0)
            error = self.computeError()
            print("\rError: {0}".format(error), end=" ", flush=True)
        return set(self.selectedInd)

    def _selectOnFeatureScore(self, threshold):
        return True

    def _selectOnDeltaScore(self, eps):
        return True
