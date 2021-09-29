
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import pytorch_lightning as pl


from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers


from deblurrer.utils.blurred_dataset import BlurredDataModule, MultipleBlurredDataModule, ConcatBlurredDataModule
from deblurrer.model.GD_simple_deblurrer import IterativeReconstructor

# for downsampling_factor 8 (steps = 3)
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

# wiener filter kappa
kappa_dict = {
    0 : 0.1, 
    1 : 0.1, 
    2 : 0.05,
    3 : 0.05, 
    4 : 0.025,
    5 : 0.025,
    6 : 0.01,
    7 : 0.007, 
    8 : 0.01,
    9 : 0.01,
    10 : 0.01,
    11 : 0.01,
    12 : 0.002,
    13 : 0.002,
    14 : 0.002,
    15 : 0.001,
    16 : 0.001,
    17 : 0.0005, 
    18 : 0.0005, 
    19 : 0.0001
}



for step in [6, 8, 9]:
#step = 2
    print("Start Training step: ", step)
    dataset = ConcatBlurredDataModule(batch_size=8, blurring_step=step)#BlurredDataModule(batch_size=8, blurring_step=step)
    dataset.prepare_data()
    dataset.setup()


    lr_monitor = LearningRateMonitor(logging_interval=None) 


    checkpoint_callback_ocr = ModelCheckpoint(
        dirpath=None,
        save_last = True,
        filename = "val_ocr_acc_{epoch}-{val_ocr_acc:.2f}", 
        save_top_k=1,
        verbose=True,
        monitor='val_ocr_acc',
        mode='max',
    )

    checkpoint_callback_loss = ModelCheckpoint(
        dirpath=None,
        save_last = False,
        filename = "val_loss_{epoch}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='max',
    )


    base_path = '/localdata/AlexanderDenker/deblurring_experiments/run_simple'
    blurring_step = "step_" + str(step)
    path_parts = [base_path, blurring_step]
    log_dir = os.path.join(*path_parts)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)

    trainer_args = {'accelerator': 'ddp',
                    'gpus': [0],
                    'default_root_dir': log_dir,
                    'callbacks': [checkpoint_callback_ocr, checkpoint_callback_loss, lr_monitor],
                    'benchmark': False,
                    'fast_dev_run': False,
                    'gradient_clip_val': 1.0,
                    'logger': tb_logger,
                    'log_every_n_steps': 10,
                    #'accumulate_grad_batches': 3, 
                    'multiple_trainloader_mode': 'min_size',
                    'auto_scale_batch_size': 'binsearch'}#,
                    #'accumulate_grad_batches': 6}#,}
                    # 'log_gpu_memory': 'all'} # might slow down performance (unnecessary uses only the output of nvidia-smi)


    reconstructor = IterativeReconstructor(radius=radius_dict[step], 
                                            n_iter=10, 
                                            downsampling_steps=3,
                                            channels=[32, 64, 128, 256], 
                                            skip_channels=[16,16,16,64], 
                                            regularization='pm', # 'pm' 
                                            use_sigmoid=False, 
                                            jittering_std=0.01, 
                                            loss='l2', 
                                            kappa_wiener=kappa_dict[step], 
                                            op_init=None)

    trainer = pl.Trainer(max_epochs=800, **trainer_args)

    trainer.fit(reconstructor, datamodule=dataset)
