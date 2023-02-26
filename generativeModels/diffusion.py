import time
import keras_cv
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

# from keras.models import load_model
print(os.path.exists("./generativeModels/images/diffusion"))
print("GPU:",tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

demo_text = "photograph of an astronaut riding a horse"

def getStableDiffusionModel():
    print("entered getStableDiffusionModel")
    model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
    print("exit getStableDiffusionModel")
    return model

# model.save('diffusion_model.h5')
# model = load_model('diffusion_model.h5')

def plot_images(images):
    print("entered plot_images")
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        ts = time.time()
        img_name = "image_" + str(ts) + str(i)
        plt.savefig("./generativeModels/images/diffusion/")
        plt.savefig("./generativeModels/images/diffusion/"+ img_name, format='png')
        plt.axis("off")
    print("exit plot_images")

def printImages(model):
    print("entered printImages")
    text = input('Enter your nft description: ')
    with tf.device('/CPU:0'):
        images = model.text_to_image(text, batch_size=1)
        plot_images(images)
    print("exit printImages")


keras.backend.clear_session()

model = getStableDiffusionModel()

# print("Model type:" + str(type(model)))
# print(str(time.time()))
# file = open("Python.txt", "w")
 
# str1 = repr(model)
# file.write(str1)
 
#close file
# file.close()

for i in range(5):
    printImages(model)

