from eyeData import EyeDataProvider
from tf_unet import unet
import platform


# nx,ny = 512, 1024

train_data_path = "/Users/miao/Downloads/eyedata/Edema_trainingset/original_images/*/*.bmp" if platform.system()=="Darwin" \
    else "/root/eyedata/Edema_trainingset/original_images/*/*.bmp"

val_data_path = "/Users/miao/Downloads/eyedata/Edema_validationset/original_images/*/*.bmp" if platform.system()=="Darwin" \
    else "/root/eyedata/Edema_validationset/original_images/*/*.bmp"

train_data_path = "/root/eyedata/Edema_trainingset/original_images/PC034_MacularCube512x128_12-10-2013_10-48-27_OS_sn14231_cube_z.img/*.bmp"
val_data_path = "/root/eyedata/Edema_validationset/original_images/PC022_MacularCube512x128_12-9-2013_15-24-2_OD_sn14152_cube_z.img/*.bmp"



train_data_provider = EyeDataProvider(train_data_path, n_class=4)
val_data_provider = EyeDataProvider(val_data_path, n_class=4)

net = unet.Unet(channels=train_data_provider.channels,
                n_class=train_data_provider.n_class,
                layers=3,
                features_root=64,
                cost_kwargs=dict(regularizer=0.001),
                cost="dice_coefficient"
                )

batch_size = 1
training_iters = int(train_data_provider.data_num/batch_size)
epochs = 50


trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(train_data_provider, "./unet_trained_eye_data",
                     val_data_provider=val_data_provider,
                     training_iters=training_iters,
                     epochs=epochs,
                     dropout=0.5,
                     display_step=1)

