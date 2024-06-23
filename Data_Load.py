import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

class data_load:
    def __init__(self):
        pass
    
    def download_data(self):
        data = tfds.load('fashion_mnist', split='train')
        keys = data.as_numpy_iterator().next().keys()
        return data, keys

    def scale_data(self, data):
        image = data['image']
        return image/255

    def print_images(self, data, n, fig_size):
        data_iterator = data.as_numpy_iterator()
        fig, ax = plt.subplots(ncols=n, figsize=(fig_size, fig_size))
        for idx in range(n):
            batch = data_iterator.next()
            ax[idx].imshow(np.squeeze(batch['image']))
            ax[idx].title.set_text(batch['label'])

if __name__ == '__main__':
    
    # Setup GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Object of class data_load
    ds = data_load()

    # Calling Member functions
    # Download data
    print("-------------START--------------")
    print("--------Downloading Data--------")
    data, keys = ds.download_data()
    print("-------Download Complete--------")
    print("-----Printing Sample Images-----")
    # Print images from the data
    ds.print_images(data, 4, 20)
    print("-------Printing Completed-------")
    print("---------Pre-Processing---------")
    # Scaling the images
    data = data.map(ds.scale_data)
    # Caching the images
    data = data.cache()
    # Shuffling the images
    data = data.shuffle(60000)
    # Putting the images in batch
    data = data.batch(128)
    # Prefetching the images
    data = data.prefetch(32)
    print("------Pre-Processing Done-------")
    print("--------Keys in the data--------")
    # Keys present in the train data
    print(keys)
    print("--------------END---------------")