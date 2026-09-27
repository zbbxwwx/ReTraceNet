"""Encode original JPEG coefficient grids; transform encoded spatial features.

No new parameters. Non-JPEG availability is unchanged. Native JPEG edge blocks
are retained; extra padding repeats whole 8x8 blocks, preserving frequency phase.
Inference must use forward_batch (not the legacy resized-volume evaluator).
"""
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from retrace.data import ManipulationDataset


def pad_blocks(coef):
    h,w=coef.shape
    assert h%8==0 and w%8==0
    blocks=coef.reshape(h//8,8,w//8,8).transpose(0,2,1,3)
    ph=(-blocks.shape[0])%4;pw=(-blocks.shape[1])%4
    blocks=np.pad(blocks,((0,ph),(0,pw),(0,0),(0,0)),mode='edge')
    return blocks.transpose(0,2,1,3).reshape((h//8+ph)*8,(w//8+pw)*8)


class NativeGridDataset(ManipulationDataset):
    @classmethod
    def from_dataset(cls, dataset):
        obj=cls.__new__(cls)
        obj.__dict__=copy.copy(dataset.__dict__)
        assert obj.dct_forensic and obj.dct_source_mode=='original_jpeg'
        return obj

    def _jpeg_info_from_original(self,record,size):
        w,h=map(int,size)
        native=torch.zeros(21,32,32)
        qt=torch.ones(1,8,8);available=torch.tensor([0.])
        if record.image.suffix.lower() in {'.jpg','.jpeg'} and record.image.exists():
            import jpegio
            try:
                jpeg=jpegio.read(str(record.image))
                coef=pad_blocks(jpeg.coef_arrays[0])
                qt=torch.from_numpy(jpeg.quant_tables[jpeg.comp_info[0].quant_tbl_no].astype('float32')[None])
                native=self._dct_volume_from_coef(coef,*coef.shape)
                available=torch.tensor([1.])
            except Exception as exc:
                raise RuntimeError(f'Native JPEG decode failed: {record.image}') from exc
        self._native=native
        nh,nw=native.shape[-2:]
        # Coordinate field goes through the EXACT legacy RGB crop-selection code.
        # It describes pixel centers in the padded original image, not coefficients.
        xx=2*(torch.arange(w,dtype=torch.float32)+.5)/nw-1
        yy=2*(torch.arange(h,dtype=torch.float32)+.5)/nh-1
        grid=torch.stack([xx[None].expand(h,w),yy[:,None].expand(h,w)])
        return grid,qt,available

    @staticmethod
    def _resize_dct(grid,size):
        assert grid.shape[0]==2, 'Only coordinates may be resized on native path'
        nw,nh=map(int,size);_,h,w=grid.shape
        # Analytic half-pixel mapping avoids edge clamping of coordinate ramps.
        dx=(grid[0,0,-1]-grid[0,0,0])/max(w-1,1)
        dy=(grid[1,-1,0]-grid[1,0,0])/max(h-1,1)
        xx=grid[0,0,0]+dx*((torch.arange(nw)+.5)*w/nw-.5)
        yy=grid[1,0,0]+dy*((torch.arange(nh)+.5)*h/nh-.5)
        return torch.stack([xx[None].expand(nh,nw),yy[:,None].expand(nh,nw)])

    def __getitem__(self,index):
        sample=super().__getitem__(index)
        sample['dct_grid']=sample['dct']
        sample['dct']=self._native
        return sample


class SeededSamples(torch.utils.data.Dataset):
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,key):
        index,seed=key
        py=random.getstate();npstate=np.random.get_state()
        try:
            random.seed(seed);np.random.seed(seed%(2**32))
            return self.dataset[index]
        finally:
            random.setstate(py);np.random.set_state(npstate)


def attach_native_grid(model):
    assert not hasattr(model,'_native_grid_handle')
    def align(branch,inputs,features):
        if not hasattr(model,'_batch_dct_grid'):
            raise RuntimeError('Native-grid model requires forward_batch and dct_grid')
        grid=model._batch_dct_grid
        h,w=model._batch_rgb_size
        outputs=[]
        for feat,stride in zip(features,[4,8,16,32]):
            target=(h//stride,w//stride)
            g=F.interpolate(grid,size=target,mode='bilinear',align_corners=False).permute(0,2,3,1)
            outputs.append(F.grid_sample(feat,g,mode='bilinear',padding_mode='border',align_corners=False))
        model._native_shapes={'raw':list(inputs[0].shape),'encoded':[list(f.shape) for f in features],
                              'aligned':[list(f.shape) for f in outputs]}
        return outputs
    model._native_grid_handle=model.dct_branch.register_forward_hook(align)


def forward_batch(model,batch):
    if hasattr(model,'_native_grid_handle'):
        model._batch_dct_grid=batch['dct_grid'].cuda(non_blocking=True)
        model._batch_rgb_size=batch['image'].shape[-2:]
    return model(**{k:batch[k].cuda(non_blocking=True) for k in ['image','dct','qtable','dct_available']})


def load_pipeline_checkpoint(path,device='cuda'):
    from retracenet.final_modules import build_final_model
    state=torch.load(path,map_location='cpu')
    model=build_final_model(state['config']['model']).to(device)
    model.load_state_dict(state['model'],strict=True)
    if state.get('dct_pipeline')=='native_grid':attach_native_grid(model)
    return model.eval(),state
