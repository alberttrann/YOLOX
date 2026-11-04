"""OPTIONAL isolated augmentation experiment for audited MosaicDetection.

Copies no weights and does not alter the original PG-SMA exp.
Install explicitly if the inherited source loader lacks MosaicDetection.
Never enable full training unless separate augment/label audits succeed.
"""
from exps.default.tde_yolox_s_pgsma import Exp as PGSMAExp


class Exp(PGSMAExp):
    def __init__(self):
        super().__init__()
        self.exp_name='tdevnext_vnext3_pgsma_mosaic_audit_ONLY'
        self.allow_full_training=False

    def get_data_loader(self,batch_size,is_distributed,no_aug=False,cache_img=False):
        from torch.utils.data import RandomSampler,DistributedSampler
        from yolox.data import COCODataset, TrainTransform, MosaicDetection
        from yolox.data import YoloBatchSampler, DataLoader as YoloDataLoader
        import torch.distributed as dist
        # get_dataset computes source COCO max annotation count and capacity.
        # This is a metadata audit input; the dataset is not used as the
        # inner MosaicDetection source because we require raw pull_item().
        source_meta=self.get_dataset(cache=False,cache_type='ram')
        del source_meta
        capacity=int(self.annotation_capacity)
        raw=COCODataset(data_dir=self.data_dir,json_file=self.train_ann,name='',
                        img_size=self.input_size,preproc=None,
                        cache=False,cache_type='ram')
        raw.cache=False;raw.cache_type='ram'
        mosaic=MosaicDetection(
            raw,img_size=self.input_size,mosaic=not no_aug,
            preproc=TrainTransform(max_labels=capacity,flip_prob=self.flip_prob,
                                  hsv_prob=self.hsv_prob),
            degrees=self.degrees,translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,shear=self.shear,
            enable_mixup=self.enable_mixup,mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        self.dataset=mosaic
        if is_distributed:
            batch_size=batch_size//dist.get_world_size()
            sampler=DistributedSampler(mosaic)
        else:
            sampler=RandomSampler(mosaic)
        batches=YoloBatchSampler(sampler=sampler,batch_size=batch_size,
                                 drop_last=False,mosaic=not no_aug)
        options={'num_workers':self.data_num_workers,'pin_memory':True,
                 'batch_sampler':batches}
        if self.data_num_workers>0:
            options['persistent_workers']=True
        return YoloDataLoader(mosaic,**options)
