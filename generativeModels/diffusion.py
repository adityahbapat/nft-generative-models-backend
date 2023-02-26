import time
import keras_cv
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

# from keras.models import load_model
print(os.path.exists("./generativeModels/images/diffusion"))
print("<.....GPU.....>:",tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('GPU'))
print("<......Num GPUs Available......>: ", len(tf.config.list_physical_devices('GPU')))


# List available physical GPUs
# gpus = tf.config.experimental.list_physical_devices('GPU')


# if gpus:
#     try:
#         for gpu in gpus:
#         # for physical GPU configure memory limit, here I configured 4GB
#             tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
#     except RuntimeError as e:
#         print(e)


# if gpus:
#     try:
#         for gpu in gpus:
#         # Allow TensorFlow to use shared GPU memory
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)


demo_text = "photograph of an astronaut riding a horse"

def getStableDiffusionModel():
    print("\n<......entered getStableDiffusionModel......>\n")
    model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
    print("\n<......exit getStableDiffusionModel......>\n")
    return model

# model.save('diffusion_model.h5')
# model = load_model('diffusion_model.h5')

def plot_images(images):
    print("\n<......entered plot_images......>\n")
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        ts = time.time()
        img_name = "image_" + str(ts) + str(i) + ".png"
        plt.savefig("./generativeModels/images/diffusion/"+ img_name)
        plt.axis("off")
    print("\n<......exit plot_images......>\n")

def printImages(model):
    print("\n<.......entered printImages.......>\n")
    text = input('Enter your NFT description: ')
    if text =="0":
        exit()
        # manually telling tf to use CPU, becaue 4GB GPU is not enough, 
        # Ran on Google Colab and saw it took more than 12GB GPU :(
    with tf.device('/CPU:0'):
        images = model.text_to_image(text, batch_size=1)
        plot_images(images)
    print("\n<.......exit printImages.......>\n")


keras.backend.clear_session()

model = getStableDiffusionModel()

for i in range(5):
    printImages(model)

