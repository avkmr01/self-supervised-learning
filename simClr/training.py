from train_logic import *
        

import torch
from pytorch_lightning import Trainer
import os
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import  resnet18
import logging
import warnings
    

def main():
    available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    save_model_path = os.path.join(os.getcwd(), "diab_ret_model/saved_models/")
    print('available_gpus:',available_gpus)
    filename=r'diab_ret_model/SimCLR_ResNet18_adam_'
    resume_from_checkpoint = False
    train_config = Hparams()

    reproducibility(train_config)
    save_name = filename + '.ckpt'

    model = SimCLR_pl(train_config, model=resnet18(weights=None), feat_dim=512)

    transform = Augment(train_config.img_size)
    data_loader = get_stl_dataloader(train_config.batch_size, transform)
    # data_path = r'./test_data'
    # data_loader = get_contrast_dataloader(data_path, train_config.batch_size, transform)

    accumulator = GradientAccumulationScheduler(scheduling={0: train_config.gradient_accumulation_steps})
    checkpoint_callback = ModelCheckpoint(filename=filename, dirpath=save_model_path,
                                            save_last=True, save_top_k=2,monitor='Contrastive loss_epoch',mode='min')

    if resume_from_checkpoint:
      trainer = Trainer(callbacks=[accumulator, checkpoint_callback],
                      accelerator='gpu', 
                      devices=2,
                      max_epochs=train_config.epochs,
                      resume_from_checkpoint=train_config.checkpoint_path)
    else:
      trainer = Trainer(callbacks=[accumulator, checkpoint_callback],
                      accelerator='gpu', 
                      devices=2,
                      max_epochs=train_config.epochs)

    trainer.fit(model, data_loader)
    trainer.save_checkpoint(save_name)
    
    model_pl = SimCLR_pl(train_config, model=resnet18(pretrained=False))
    model_pl = weights_update(model_pl, r"diab_ret_model/SimCLR_ResNet18_adam_.ckpt")

    resnet18_backbone_weights = model_pl.model.backbone
    # print(resnet18_backbone_weights)
    torch.save({
                'model_state_dict': resnet18_backbone_weights.state_dict(),
                }, r'diab_ret_model/resnet18_backbone_weights.ckpt')

if __name__ == '__main__':
    # logging.getLogger("pl_bolts").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    main()