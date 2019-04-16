import keras as keras
from keras import backend as K


class CalculateInformationCallback(keras.callbacks.Callback):
    """
    Calls informationProcessor to calculate mutual information per batch
    """

    def __init__(self, model, information_processor, x_test):
        super().__init__()
        outputs = [layer.output for layer in model.layers]
        self.batch = 0
        self.__functor = K.function([model.input, K.learning_phase()], outputs)
        self.__x_test = x_test
        self.__ip = information_processor

    def on_batch_end(self, batch, logs=None):
        self.batch += 1

        def activation():
            return self.__functor([self.__x_test, 0.])
        self.__ip.calculate_information(activation, self.batch)

    def on_train_end(self, logs=None):
        self.__ip.finish_information_calculation()
