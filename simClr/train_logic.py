import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import SGD, Adam
import torch.nn as nn
from utils import *


class AddProjection(nn.Module):
    def __init__(self, config, model=None, mlp_dim=512):
        super(AddProjection, self).__init__()
        embedding_size = config.embedding_size
        self.backbone = default(model, models.resnet18(weights=None, num_classes=config.embedding_size))
        mlp_dim = default(mlp_dim, self.backbone.fc.in_features)
        print('Dim MLP input:',mlp_dim)
        self.backbone.fc = nn.Identity()

        # add mlp projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
        embedding = self.backbone(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)



def define_param_groups(model, weight_decay, optimizer_name):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if optimizer_name == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    return param_groups


class SimCLR_pl(pl.LightningModule):
    def __init__(self, config, model=None, feat_dim=512):
        super().__init__()
        self.config = config
        
        self.model = AddProjection(config, model=model, mlp_dim=feat_dim)

        self.loss = ContrastiveLoss(config.batch_size, temperature=self.config.temperature)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        (x1, x2), labels = batch
        z1 = self.model(x1)
        z2 = self.model(x2)
        loss = self.loss(z1, z2)
        self.log('Contrastive loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        max_epochs = int(self.config.epochs)
        param_groups = define_param_groups(self.model, self.config.weight_decay, 'adam')
        lr = self.config.lr
        optimizer = Adam(param_groups, lr=lr, weight_decay=self.config.weight_decay)

        print(f'Optimizer Adam, '
              f'Learning Rate {lr}, '
              f'Effective batch size {self.config.batch_size * self.config.gradient_accumulation_steps}')

        scheduler_warmup = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=max_epochs,
                                                         warmup_start_lr=0.0)

        return [optimizer], [scheduler_warmup]
    
    
# a lazy way to pass the config file
class Hparams:
    def __init__(self):
        self.epochs = 100 # number of training epochs
        self.seed = 77777 # randomness seed
        self.cuda = True # use nvidia gpu
        self.img_size = 96 #image shape
        self.save = "./diab_ret_model/saved_models/" # save checkpoint
        self.load = False # load pretrained checkpoint
        self.gradient_accumulation_steps = 5 # gradient accumulation steps
        self.batch_size = 35
        self.lr = 3e-4 # for ADAm only
        self.weight_decay = 1e-6
        self.embedding_size= 128 # papers value is 128
        self.temperature = 0.5 # 0.1 or 0.5
        self.checkpoint_path = '.diab_ret_model/SimCLR_ResNet18.ckpt' # replace checkpoint path here