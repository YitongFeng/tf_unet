from eyeData import EyeDataProvider
from tf_unet import unet
import platform
from tf_unet import image_gen




train_data_provider = image_gen.GrayScaleDataProvider(572, 572, cnt=20)
val_data_provider = image_gen.GrayScaleDataProvider(572, 572, cnt=20)

net = unet.Unet(channels=train_data_provider.channels,
                n_class=train_data_provider.n_class,
                layers=3,
                features_root=16,
                cost_kwargs=dict(regularizer=0.001),
                # cost="dice_coefficient"
                )

batch_size = 1
training_iters = 32
epochs = 10


trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(train_data_provider, "./unet_trained_toy_data",
                     val_data_provider=val_data_provider,
                     training_iters=training_iters,
                     epochs=epochs,
                     dropout=0.5,
                     display_step=1)

