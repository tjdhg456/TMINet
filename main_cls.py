import os
from pytorch_lightning.loggers.neptune import NeptuneLogger
import pytorch_lightning as pl

from trainer import disguise_GAN
from utils_cls import init_args_cls

import random
import warnings
warnings.filterwarnings(action='ignore')

# Main
if __name__ == '__main__':
    args = init_args_cls()

    neptune_logger = NeptuneLogger(
        api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzdlYWFkMjctOWExMS00YTRlLWI0MWMtY2FhNmIyNzZlYTIyIn0=',
        project_name="sunghoshin/OBJECT-CLS",
        close_after_fit=False,
        experiment_name='%s_%d_%s' % (args.exp_name, args.exp_num, args.server),
        params={"max_epochs": args.epoch_num,
                "batch_size": args.batch_size,
                "lr_g": args.lr_g,
                "lr_d": args.lr_d},
        tags=['data:%s' % args.network_type, 'resume:%s' %args.resume, \
              'model_type:%s' % args.model_type, 'G_name:%s' %args.G_name, \
              'lambda_c:%s' %str(args.lambda_c), 'lambda_g:%s' %str(args.lambda_g),\
              'lambda_d:%s' %str(args.lambda_d), \
              'exp_name:%s' %args.exp_name, 'exp_num:%s' %args.exp_num, \
              'server:%s' %args.server, 'privacy:%s' %args.privacy, \
              'pretrained:%s' %args.pretrained],
        offline_mode=False
    )

    # Callback
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=args.save_folder,
                                                    save_top_k=1,
                                                    monitor='val_top1',
                                                    mode='max',
                                                    save_last=True)

    # Trainer
    trainer = pl.Trainer(gpus=args.gpu,
                         distributed_backend='ddp',
                         max_epochs=args.epoch_num,
                         logger=neptune_logger,
                         # early_stop_callback=early_stop_callback,
                         checkpoint_callback=model_checkpoint, )

    # Fit
    if args.privacy:
        model_trainer = disguise_GAN(args)

    # Load Classification model
    if args.resume:
        print('Resume the pre-trained network')
        model_trainer = model_trainer.load_from_checkpoint(args.path)
    elif os.path.isfile(args.resume_path):
        print('Load the pre-trained network')
        model_trainer = model_trainer.load_from_checkpoint(args.resume_path)

    # Start
    trainer.fit(model_trainer)

    # Finish
    neptune_logger.experiment.stop()  # STOP!