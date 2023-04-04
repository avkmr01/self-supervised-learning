from train_logic import *
from finetuning import *

def main():
    available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    train_config = Hparams()
    # load model
    save_model_path = os.path.join(os.getcwd(), "saved_models/")
    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Identity()
    print('imagenet weights, no pretraining')
    model = SimCLR_eval(train_config.lr, model=resnet, linear_eval=False)
    filename = 'direct_'
    # preprocessing and data loaders
    transform_preprocess = Augment(train_config.img_size).test_transform
    save_name = filename + 'train.ckpt'

    data_loader = get_stl_dataloader(128, transform=transform_preprocess,split='train')
    data_loader_test = get_stl_dataloader(128, transform=transform_preprocess,split='test')

    checkpoint_callback = ModelCheckpoint(filename=filename, dirpath=save_model_path)

    trainer = Trainer(callbacks=[checkpoint_callback],
                      gpus=available_gpus,
                      max_epochs=train_config.epochs)

    trainer.fit(model, data_loader, data_loader_test)
    trainer.save_checkpoint(save_name)
    
if __name__ == '__main__':
    main()