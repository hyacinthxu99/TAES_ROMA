"""AUC, VUS and F1 metrics used in the experiments. F1 uses oracle thresholds."""
from sklearn import metrics
import numpy as np
import copy

def get_metrics(score, labels, slidingWindow=60, pred=None, version='opt', thre=100):
    metrics = {}
    grader = basic_metricor()
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)
    (_, _, _, _, _, _, VUS_ROC, VUS_PR) = generate_curve(labels, score, slidingWindow, version, thre)
    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    metrics['AUC-PR'] = AUC_PR
    metrics['AUC-ROC'] = AUC_ROC
    metrics['VUS-PR'] = VUS_PR
    metrics['VUS-ROC'] = VUS_ROC
    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    return metrics

def generate_curve(label, score, slidingWindow, version='opt', thre=250):
    if version == 'opt_mem':
        (tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d) = basic_metricor().RangeAUC_volume_opt_mem(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
    else:
        (tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d) = basic_metricor().RangeAUC_volume_opt(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
    X = np.array(tpr_3d).reshape(1, -1).ravel()
    X_ap = np.array(tpr_3d)[:, :-1].reshape(1, -1).ravel()
    Y = np.array(fpr_3d).reshape(1, -1).ravel()
    W = np.array(prec_3d).reshape(1, -1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0]) - 1)
    return (Y, Z, X, X_ap, W, Z_ap, avg_auc_3d, avg_ap_3d)

class basic_metricor:

    def __init__(self, a=1, probability=True, bias='flat'):
        self.a = a
        self.probability = probability
        self.bias = bias
        self.eps = 1e-15

    def _adjust_predicts(self, score, label, threshold=None, pred=None, calc_latency=False):
        if len(score) != len(label):
            raise ValueError('score and label must have the same length')
        score = np.asarray(score)
        label = np.asarray(label)
        latency = 0
        if pred is None:
            predict = score > threshold
        else:
            predict = copy.deepcopy(pred)
        actual = label > 0.1
        anomaly_state = False
        anomaly_count = 0
        for i in range(len(score)):
            if actual[i] and predict[i] and (not anomaly_state):
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    elif not predict[j]:
                        predict[j] = True
                        latency += 1
            elif not actual[i]:
                anomaly_state = False
            if anomaly_state:
                predict[i] = True
        if calc_latency:
            return (predict, latency / (anomaly_count + 0.0001))
        else:
            return predict

    def metric_ROC(self, label, score):
        return metrics.roc_auc_score(label, score)

    def metric_PR(self, label, score):
        return metrics.average_precision_score(label, score)

    def metric_PointF1(self, label, score, preds=None):
        if preds is None:
            (precision, recall, thresholds) = metrics.precision_recall_curve(label, score)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-05)
            F1 = np.max(f1_scores)
            threshold = thresholds[np.argmax(f1_scores)]
        else:
            (Precision, Recall, F, Support) = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
            F1 = F[1]
        return F1

    def metric_PointF1PA(self, label, score, preds=None):
        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            PointF1PA_scores = []
            for threshold in thresholds:
                preds = (score > threshold).astype(int)
                adjust_preds = self._adjust_predicts(score, label, pred=preds)
                PointF1PA = metrics.f1_score(label, adjust_preds)
                PointF1PA_scores.append(PointF1PA)
            PointF1PA_Threshold = thresholds[np.argmax(PointF1PA_scores)]
            PointF1PA1 = max(PointF1PA_scores)
        else:
            adjust_preds = self._adjust_predicts(score, label, pred=preds)
            PointF1PA1 = metrics.f1_score(label, adjust_preds)
        return PointF1PA1

    def _get_events(self, y_test, outlier=1, normal=0):
        events = dict()
        label_prev = normal
        event = 0
        event_start = 0
        for (tim, label) in enumerate(y_test):
            if label == outlier:
                if label_prev == normal:
                    event += 1
                    event_start = tim
            elif label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
            label_prev = label
        if label_prev == outlier:
            event_end = tim - 1
            events[event] = (event_start, event_end)
        return events

    def metric_EventF1PA(self, label, score, preds=None):
        from sklearn.metrics import precision_score
        true_events = self._get_events(label)
        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            EventF1PA_scores = []
            for threshold in thresholds:
                preds = (score > threshold).astype(int)
                tp = np.sum([preds[start:end + 1].any() for (start, end) in true_events.values()])
                fn = len(true_events) - tp
                rec_e = tp / (tp + fn)
                prec_t = precision_score(label, preds)
                EventF1PA = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)
                EventF1PA_scores.append(EventF1PA)
            EventF1PA_Threshold = thresholds[np.argmax(EventF1PA_scores)]
            EventF1PA1 = max(EventF1PA_scores)
        else:
            tp = np.sum([preds[start:end + 1].any() for (start, end) in true_events.values()])
            fn = len(true_events) - tp
            rec_e = tp / (tp + fn)
            prec_t = precision_score(label, preds)
            EventF1PA1 = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)
        return EventF1PA1

    def range_convers_new(self, label):
        anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
        (anomaly_ends,) = np.where(np.diff(label) == -1)
        if len(anomaly_ends):
            if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
                anomaly_starts = np.concatenate([[0], anomaly_starts])
        if len(anomaly_starts):
            if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
                anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
        return list(zip(anomaly_starts, anomaly_ends))

    def new_sequence(self, label, sequence_original, window):
        a = max(sequence_original[0][0] - window // 2, 0)
        sequence_new = []
        for i in range(len(sequence_original) - 1):
            if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
                sequence_new.append((a, sequence_original[i][1] + window // 2))
                a = sequence_original[i + 1][0] - window // 2
        sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
        return sequence_new

    def sequencing(self, x, L, window=5):
        label = x.copy().astype(float)
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]
            x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
            label[x1] += np.sqrt(1 - (x1 - e) / window)
            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / window)
        label = np.minimum(np.ones(length), label)
        return label

    def RangeAUC_volume_opt(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)
        score_sorted = -np.sort(-score)
        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))
        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)
        tp = np.zeros(thre)
        N_pred = np.zeros(thre)
        for (k, i) in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            N_pred[k] = np.sum(pred)
        for window in window_3d:
            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)
            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0
            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                threshold = score_sorted[i]
                pred = score >= threshold
                labels = labels_extended.copy()
                existence = 0
                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                    if (pred[seg[0]:seg[1] + 1] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1
                TP = 0
                N_labels = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])
                TP += tp[j]
                FP = N_pred[j] - TP
                existence_ratio = existence / len(L)
                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)
                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new
                Precision = TP / N_pred[j]
                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision
            TF_list[j + 1] = [1, 1]
            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list
            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = AUC_range
            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]
            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range
        return (tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d))

    def RangeAUC_volume_opt_mem(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)
        score_sorted = -np.sort(-score)
        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))
        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)
        tp = np.zeros(thre)
        N_pred = np.zeros(thre)
        p = np.zeros((thre, len(score)))
        for (k, i) in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            p[k] = pred
            N_pred[k] = np.sum(pred)
        for window in window_3d:
            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)
            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0
            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                labels = labels_extended.copy()
                existence = 0
                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * p[j][seg[0]:seg[1] + 1]
                    if (p[j][seg[0]:seg[1] + 1] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1
                N_labels = 0
                TP = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], p[j][seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])
                TP += tp[j]
                FP = N_pred[j] - TP
                existence_ratio = existence / len(L)
                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)
                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new
                Precision = TP / N_pred[j]
                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision
            TF_list[j + 1] = [1, 1]
            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list
            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = AUC_range
            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]
            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range
        return (tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d))
