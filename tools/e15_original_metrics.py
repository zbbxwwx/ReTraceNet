"""Shared original-GT-grid validation accumulator (no test selection)."""
import numpy as np


class PixelStats:
    def __init__(self, thresholds=101, bins=65536):
        self.thresholds = np.linspace(0, 1, thresholds, dtype=np.float64)
        self.pos = np.zeros(thresholds, dtype=np.int64)
        self.neg = np.zeros(thresholds, dtype=np.int64)
        self.auc_pos = np.zeros(bins, dtype=np.int64)
        self.auc_neg = np.zeros(bins, dtype=np.int64)
        self.count = 0

    def add(self, scores, target):
        s = np.asarray(scores, dtype=np.float32).reshape(-1)
        y = np.asarray(target).reshape(-1)
        assert s.shape == y.shape
        valid = y >= 0
        s, y = s[valid], y[valid] > 0
        assert np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()
        index = np.searchsorted(self.thresholds, s, side='right') - 1
        self.pos += np.bincount(index[y], minlength=len(self.pos))
        self.neg += np.bincount(index[~y], minlength=len(self.neg))
        ai = (s * (len(self.auc_pos) - 1)).astype(np.int64)
        self.auc_pos += np.bincount(ai[y], minlength=len(self.auc_pos))
        self.auc_neg += np.bincount(ai[~y], minlength=len(self.auc_neg))
        self.count += 1

    def result(self):
        tp = self.pos[::-1].cumsum()[::-1]
        fp = self.neg[::-1].cumsum()[::-1]
        fn, tn = self.pos.sum() - tp, self.neg.sum() - fp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        def at(i):
            return {'threshold': float(self.thresholds[i]), 'tp': int(tp[i]), 'fp': int(fp[i]),
                    'fn': int(fn[i]), 'tn': int(tn[i]), 'f1': float(f1[i]),
                    'precision': float(tp[i] / max(tp[i] + fp[i], 1)),
                    'recall': float(tp[i] / max(tp[i] + fn[i], 1)),
                    'iou': float(tp[i] / max(tp[i] + fp[i] + fn[i], 1)),
                    'fpr': float(fp[i] / max(fp[i] + tn[i], 1))}
        total_pos, total_neg = int(self.auc_pos.sum()), int(self.auc_neg.sum())
        lower_neg = self.auc_neg.cumsum() - self.auc_neg
        numerator = (self.auc_pos.astype(np.float64) * (lower_neg + .5 * self.auc_neg)).sum()
        auc = float(numerator / (total_pos * total_neg)) if total_pos and total_neg else None
        return {'count': self.count, 'pixels': total_pos + total_neg, 'auc': auc,
                'auc_method': '65536_bin_pairwise_ties_half',
                'fixed': at(int(np.abs(self.thresholds - .5).argmin())),
                'oracle': at(int(f1.argmax()))}


def stratum(path, target):
    name = str(path).split('/')[-1]
    kind = 'Tp_S' if name.startswith('Tp_S') else ('Tp_D' if name.startswith('Tp_D') else 'unknown')
    y = np.asarray(target)
    valid = y >= 0
    ratio = float((y[valid] > 0).mean())
    area = 'lt_1' if ratio < .01 else ('1_5' if ratio < .05 else ('5_20' if ratio < .2 else 'gte_20'))
    return kind + ':' + area


class ValidationStats:
    def __init__(self):
        self.groups = {'all': PixelStats(), 'fake': PixelStats()}

    def add(self, scores, target, label, path):
        self.groups['all'].add(scores, target)
        if label == 1 and (np.asarray(target) > 0).any():
            self.groups['fake'].add(scores, target)
            key = stratum(path, target)
            self.groups.setdefault(key, PixelStats()).add(scores, target)

    def result(self):
        return {name: value.result() for name, value in self.groups.items()}
