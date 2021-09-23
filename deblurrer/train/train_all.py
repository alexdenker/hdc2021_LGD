
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pytorch_lightning as pl


from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers


from deblurrer.utils.blurred_dataset import BlurredDataModule, MultipleBlurredDataModule
from deblurrer.model.GD_deblurrer_downsampling import IterativeReconstructor

radius_dict = {
    0 : 1.0, 
    1 : 1.2, 
    2 : 1.3,
    3 : 1.4, 
    4 : 2.2,
    5 : 3.75,
    6 : 4.5,
    7 : 5.25, 
    8 : 6.75,
    9 : 8.2,
    10 : 8.8,
    11 : 9.4,
    12 : 10.3,
    13 : 10.8,
    14 : 11.5,
    15 : 12.1,
    16 : 13.5,
    17 : 16., 
    18 : 17.8, 
    19 : 19.4
}

for step in range(17, 20):
#step = 2
    print("Start Training step: ", step)
    dataset = MultipleBlurredDataModule(batch_size=8, blurring_step=step)#BlurredDataModule(batch_size=8, blurring_step=step)
    dataset.prepare_data()
    dataset.setup()


    lr_monitor = LearningRateMonitor(logging_interval=None) 


    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        save_last = True, 
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    base_path = '/localdata/AlexanderDenker/deblurring_experiments'
    experiment_name = 'pm_no_sigmoid'
    blurring_step = "step_" + str(step)
    path_parts = [base_path, experiment_name, blurring_step]
    log_dir = os.path.join(*path_parts)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)

    trainer_args = {'accelerator': 'ddp',
                    'gpus': [0],
                    'default_root_dir': log_dir,
                    'callbacks': [checkpoint_callback, lr_monitor],
                    'benchmark': False,
                    'fast_dev_run': False,
                    'gradient_clip_val': 1.0,
                    'logger': tb_logger,
                    'log_every_n_steps': 10,
                    #'accumulate_grad_batches': 4, 
                    'multiple_trainloader_mode': 'min_size',
                    'auto_scale_batch_size': 'binsearch'}#,
                    #'accumulate_grad_batches': 6}#,}
                    # 'log_gpu_memory': 'all'} # might slow down performance (unnecessary uses only the output of nvidia-smi)


    reconstructor = IterativeReconstructor(radius=radius_dict[step], n_memory=5, n_iter=11, channels=[32,64, 128, 128], skip_channels=[16,32,32,64], img_shape=(181, 294), regularization='pm', use_sigmoid=False)

    trainer = pl.Trainer(max_epochs=300, **trainer_args)

    trainer.fit(reconstructor, datamodule=dataset)
