"""One protocol for V3 validation, cross-dataset, and in-domain test reporting."""
import numpy as np

class PixelCounts:
    def __init__(self, threshold):
        self.threshold=float(threshold); self.tp=self.fp=self.fn=self.tn=0
    def add(self,scores,target):
        pred=np.asarray(scores)>=self.threshold; truth=np.asarray(target)>0
        self.tp+=int(np.logical_and(pred,truth).sum()); self.fp+=int(np.logical_and(pred,~truth).sum())
        self.fn+=int(np.logical_and(~pred,truth).sum()); self.tn+=int(np.logical_and(~pred,~truth).sum())
    def result(self):
        tp,fp,fn,tn=self.tp,self.fp,self.fn,self.tn
        return {'threshold':self.threshold,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
          'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),
          'iou':tp/max(tp+fp+fn,1),'fpr':fp/max(fp+tn,1)}

class AuthenticPixelFPR:
    def __init__(self,bins=65536): self.bins=bins; self.hist=np.zeros(bins,dtype=np.int64)
    def add(self,scores):
        idx=np.minimum((np.clip(np.asarray(scores),0,1)*(self.bins-1)).astype(np.int64),self.bins-1)
        self.hist += np.bincount(idx.ravel(),minlength=self.bins)
    def result(self,threshold):
        first=int(np.ceil(float(threshold)*(self.bins-1))); first=max(0,min(first,self.bins))
        fp=int(self.hist[first:].sum()); total=int(self.hist.sum())
        return {'threshold':float(threshold),'false_positive_pixels':fp,'authentic_pixels':total,'fpr':fp/max(total,1)}

class ImageDetectionStats:
    def __init__(self): self.scores=[]; self.labels=[]
    def add(self,score,label): self.scores.append(float(score)); self.labels.append(int(label))
    @staticmethod
    def _counts(scores,labels,t):
        c=PixelCounts(t); c.add(scores,labels); return c.result()
    def result(self,locked_threshold=None):
        s=np.asarray(self.scores,dtype=np.float64); y=np.asarray(self.labels,dtype=np.int16)
        out={'count':int(y.size),'fake_count':int(y.sum()),'authentic_count':int((y==0).sum())}
        if y.size==0: return out
        out['fixed_0p5_diagnostic']=self._counts(s,y,.5)
        if locked_threshold is not None: out['test_f1_locked']=self._counts(s,y,float(locked_threshold))
        grid=np.linspace(0,1,1001); vals=[self._counts(s,y,t) for t in grid]
        out['test_f1_oracle_diagnostic']=max(vals,key=lambda x:x['f1'])
        if len(np.unique(y))==2:
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                out['auc']=float(roc_auc_score(y,s)); out['average_precision']=float(average_precision_score(y,s))
            except Exception as ex: out['auc']=None; out['average_precision']=None; out['metric_error']=repr(ex)
        else: out['auc']=None; out['average_precision']=None
        return out

PROTOCOL={
 'name':'V3_30epoch_unified_protocol',
 'pixel_grid':'original image / GT grid; padding excluded',
 'probability_resize':'bilinear probability interpolation when alignment is required',
 'valid_pixels':'GT >= 0',
 'pixel_primary_population':'fake images only',
 'authentic_pixel_reporting':'authentic images excluded from localization F1 and reported separately as pixel FPR',
 'image_detection_population':'all fake and authentic images',
 'test_threshold':'locked from the selected checkpoint validation fake-only pooled F1 optimum',
 'oracle_policy':'test oracle is diagnostic only and never used for selection or the primary test result',
}
