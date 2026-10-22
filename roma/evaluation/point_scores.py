import numpy as np

class PointScores:
    """Average overlapping reconstruction scores without storing repeated labels."""
    def __init__(self, rows, stride):
        self.total = np.zeros(rows, dtype=np.float64)
        self.count = np.zeros(rows, dtype=np.int32)
        self.stride = stride
        self.windows = 0

    def add(self, scores):
        scores = np.asarray(scores)
        starts = (self.windows + np.arange(len(scores))) * self.stride
        positions = starts[:, None] + np.arange(scores.shape[1])
        if positions.size and positions.max() >= len(self.total):
            raise ValueError('ESA score windows exceed the test timeline')
        np.add.at(self.total, positions.ravel(), scores.ravel())
        np.add.at(self.count, positions.ravel(), 1)
        self.windows += len(scores)

    def finish(self, labels, evaluation_mask):
        valid = np.asarray(evaluation_mask, dtype=bool) & (self.count > 0)
        scores = self.total[valid] / self.count[valid]
        labels = np.asarray(labels).reshape(-1)[valid]
        if len(np.unique(labels)) != 2:
            raise ValueError('Selected ESA channels have no evaluable anomaly/normal pair')
        return scores, labels, valid
