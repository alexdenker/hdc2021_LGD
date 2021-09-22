
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import pytorch_lightning as pl


from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers


from deblurrer.utils.blurred_dataset import BlurredDataModule, MultipleBlurredDataModule
from deblurrer.model.GD_deblurrer_downsampling import IterativeReconstructor

step = 15


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

channels_list = [[4,8, 16, 32], [8,16, 32, 64], [16,32, 64, 128], [32,64, 64, 128]]
skip_list = [[4,4, 8, 16], [8,8, 16, 32], [16,16, 32, 64], [32,32, 64, 128]]
n_memory = [1, 2, 3, 4, 5]
n_iter = [4,6,8,10, 12]
regularization = ['pm', None]

for i in range(len(regularization)):
    for j in range(len(n_iter)):
        for k in range(len(n_memory)):
            for l in range(len(channels_list)):


                print("Start Training step: ", step)
                dataset = MultipleBlurredDataModule(batch_size=8, blurring_step=step)#BlurredDataModule(batch_size=8, blurring_step=step)
                dataset.prepare_data()
                dataset.setup()

                checkpoint_callback = ModelCheckpoint(
                    dirpath=None,
                    filename='learned_gradient_descent',
                    save_top_k=1,
                    verbose=True,
                    monitor='val_loss',
                    mode='min',
                )

                base_path = '/localdata/AlexanderDenker/deblurring_experiments'
                experiment_name = 'modifications'
                blurring_step = "step_" + str(step)
                path_parts = [base_path, experiment_name, blurring_step]
                log_dir = os.path.join(*path_parts)
                tb_logger = pl_loggers.TensorBoardLogger(log_dir)

                trainer_args = {'accelerator': 'ddp',
                    'gpus': [0],
                    'default_root_dir': log_dir,
                    'callbacks': [checkpoint_callback],
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


                reconstructor = IterativeReconstructor(radius=radius_dict[step], 
                                                        n_memory=n_memory[k], 
                                                        n_iter=n_iter[j], 
                                                        channels=channels_list[l], 
                                                        skip_channels=skip_list[l], 
                                                        img_shape=(181, 294), 
                                                        regularization=regularization[i], 
                                                        use_sigmoid=False)

                trainer = pl.Trainer(max_epochs=200, **trainer_args)

                trainer.fit(reconstructor, datamodule=dataset)
