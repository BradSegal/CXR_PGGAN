import pytorch_lightning as pl

from utility import Checkpoint


def GAN_trainer(GAN_model, ngpu, iter_per_save=500, filepath="./Checkpoints/"):
    checkpoint_callback = Checkpoint(filepath, save_rate=iter_per_save)
    total_epochs = 1
    GANTrainer = pl.Trainer(gpus=ngpu, min_epochs=total_epochs, max_epochs=total_epochs,
                            callbacks=[checkpoint_callback], distributed_backend='dp', early_stop_callback=False,
                            reload_dataloaders_every_epoch=True, profiler=True)

    GANTrainer.progress_bar_refresh_rate = 1
    GANTrainer.fit(GAN_model)
