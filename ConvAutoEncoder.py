import os
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from PIL import Image
from Handler import Handler
from keras.models import load_model
tamanho =52


class ConvAutoEncoder:

    def __init__(self, input_shape, output_dim, filters=None,
                 kernel=(3, 3), optimizer='adadelta', lossfn='mean_squared_error'):# mean_squared_error sparse_categorical_crossentropy categorical_crossentropy

        if filters is None:
            filters = [16, 8, 8]

        self.mse = None

        self.input_shape = input_shape
        self.output_dim = output_dim

        input_layer = Input(input_shape)

        x = Conv2D(filters[0], kernel, activation='relu', padding='same')(input_layer)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(filters[1], kernel, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(filters[2], kernel, activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(filters[2], kernel, activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters[1], kernel, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters[0], kernel, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, kernel, activation='sigmoid', padding='same')(x)

        # create autoencoder and decoder model
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)

        # create a placeholder for an encoded input
        enc_shape = encoded[0].shape

        encoded_input = Input(shape=(int(enc_shape[0]), int(enc_shape[1]), int(enc_shape[2])))

        # retrieve the decoder layers and apply to each prev layer
        num_decoder_layers = 7
        decoder_layer = encoded_input
        for i in range(-num_decoder_layers, 0):
            decoder_layer = self.autoencoder.layers[i](decoder_layer)

        # create the decoder model
        self.decoder = Model(encoded_input, decoder_layer)

        # compile model
        self.autoencoder.compile(optimizer=optimizer, loss=lossfn)

        self.autoencoder.summary()

    def fit(self, train, test, epochs=50, batch_size=None, shuffle=True, validation_data=None, callbacks=None):

        if callbacks is None:
            callbacks = [TensorBoard(log_dir='/tmp/autoencoder')]

        fit_return = self.autoencoder.fit(x=train, y=train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                                          validation_data=validation_data, callbacks=callbacks)

        self.mse = self.autoencoder.evaluate(test, test)
        print('CAE MSE on validation data: ', self.mse)

        return fit_return

    def encode(self, input):
        return self.encoder.predict(input)

    def decode(self, codes):
        return self.decoder.predict(codes)

    def autoencode(self, input):
        return self.autoencoder.predict(input)

    def history(self):
        return self.autoencoder.history

    def save_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.autoencoder.save_weights(os.path.join(path, prefix + "autoencoder_weights.h5"))
        self.autoencoder.save(os.path.join(path, prefix + "autoencoder_model.h5"))
        self.encoder.save_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.encoder.save(os.path.join(path, prefix + "encoder_model.h5"))
        self.decoder.save_weights(os.path.join(path, prefix + "decoder_weights.h5"))
        self.decoder.save(os.path.join(path, prefix + "decoder_model.h5"))

    def load_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.autoencoder.load_weights(os.path.join(path, prefix + "autoencoder_weights.h5"))
        self.encoder.load_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.load_weights(os.path.join(path, prefix + "decoder_weights.h5"))


def configureDataset():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), tamanho, tamanho, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), tamanho, tamanho, 1))  # adapt this if using `channels_first` image data format

    return x_train, x_test


def save_img(imgs, stri):
    num = imgs.shape[0]

    for i in range(num):
        A = imgs[i].copy() * 255
        A = np.reshape(np.ravel(A), (tamanho, tamanho))
        new_p = Image.fromarray(A)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(os.path.join(stri, str(i) + ".jpg"))

def configure_dataset():
    Handler().write_datafile()

    fist, last = 50, 150

    img, classify = Handler().read_datafile()

    img_test = np.asarray([img[i] for i in range(fist, last)])
    classfi_test = np.asarray([classify[i] for i in range(fist, last)])

    for i in range(last, fist-1, -1):
        np.delete(img, i)
        np.delete(classify, i)

    img, classify, img_test, classfi_test = img.astype('float32'), np.asarray(classify).astype('float32'), img_test.astype('float32'), classfi_test.astype('float32')
    return img, np.asarray(classify), img_test, classfi_test

if __name__ == '__main__':
    #x_train, x_test = configureDataset()
    img_train, classify_train, img_test, classify_test = configure_dataset()
    #save_img(x_test, 'IN/')

    #auto = ConvAutoEncoder(img_train[0].shape, img_train[0].shape, filters=[8,8,8])

    print(img_train.shape, img_train.shape)

    #auto.fit(x_train, x_test, epochs=1, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

    #auto.save_weights(prefix="mnist_")

    #auto.load_weights(prefix="db_08-08-08_")

    #a = auto.autoencode(img_test)

    model = load_model('db_08-08-08_autoencoder_model.h5')

    a = model.predict(img_test)

    save_img(a, 'OUT/')
