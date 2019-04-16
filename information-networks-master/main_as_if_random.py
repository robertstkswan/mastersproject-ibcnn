import keras
import numpy as np
from information.NaftaliTishby import __conditional_entropy, entropy_of_data
from keras import backend as K
from utils import bin_array, hash_data
import _pickle
import argparse
import math
from networks.networks import network_parameters, get_model_categorical
from data.data import load_data, parameters_data
from plot.plot import plot_one


def main():
    args = get_parameters()

    (x_train, y_train), (x_test, y_test), categories = load_data(args.data_set, args.train_size)

    model = get_model_categorical(
        input_shape=x_train[0].shape,
        network_shape=args.shape,
        categories=categories,
        activation=args.activation)
    no_of_batches = math.ceil(len(x_train) / args.batch_size) * args.epochs
    save_layers_callback = SaveLayers(model, x_test, max(no_of_batches - args.no_saved_epochs, 0))
    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              callbacks=[save_layers_callback],
              epochs=args.epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    saved = save_layers_callback.saved_layers

    print("data_x")
    x_test_hash = hash_data(x_test)
    data_x = x_test_hash
    for _ in range(args.no_saved_epochs - 1):
        data_x = np.concatenate((data_x, x_test_hash))

    print("data_y")
    y_test_hash = hash_data(y_test)
    data_y = y_test_hash
    for _ in range(args.no_saved_epochs - 1):
        data_y = np.concatenate((data_y, y_test_hash))

    print("data_t")
    # saved data where every number is binned
    saved_bin = [[bin_array(layer, bins=args.bins, low=layer.min(), high=layer.max()) for layer in epoch] for epoch in saved]
    # saved data where every number is hashed
    saved_hash = [[hash_data(layer) for layer in epoch] for epoch in saved_bin]

    data_t = {}
    for t in range(len(saved_hash[0])):
        data_t[t] = np.array([], dtype=np.int64)
    for epoch in range(len(saved_hash)):
        for t in range(len(saved_hash[0])):
            data_t[t] = np.concatenate([data_t[t], saved_hash[epoch][t]])
    data_t = list(data_t.values())

    print("entropy")
    h_t = np.array([entropy_of_data(t) for t in data_t])
    h_t_x = np.array([__conditional_entropy(t, data_x) for t in data_t])
    h_t_y = np.array([__conditional_entropy(t, data_y) for t in data_t])

    i_x_t = h_t - h_t_x
    i_y_t = h_t - h_t_y

    path = args.dest + "/data/as_if_random/" + filename(args)
    _pickle.dump((i_x_t, i_y_t), open(path, 'wb'))
    path = args.dest + "/images/as_if_random/" + filename(args)
    plot_one(i_x_t, i_y_t, filename=path)

    print(i_x_t)
    print(i_y_t)


    return


def get_probabilities(data):
    unique_array, unique_counts = np.unique(data, return_counts=True)
    prob = unique_counts / np.sum(unique_counts)
    return dict(zip(unique_array, prob))


def get_probabilities_map(map_data):
    prob = {}
    for key in map_data:
        prob[key] = get_probabilities(map_data[key])
    return prob


class SaveLayers(keras.callbacks.Callback):

    def __init__(self, model, x_test, save_layer=0):
        super().__init__()
        outputs = [layer.output for layer in model.layers]
        if type(save_layer) is int:
            self.__save_layer = lambda x: x > save_layer
        else:
            # assumes save_layer is a function
            self.__save_layer = save_layer

        # columns = ["x", "y"] + list(map(lambda x: "t" + str(x+1), list(range(len(model.layers)))))
        # self.saved_layers = pd.DataFrame(columns=columns)
        self.saved_layers = []
        self.__row_id = 0

        self.__batch = 0
        self.__functor = K.function([model.input, K.learning_phase()], outputs)
        self.__x_test = x_test

    def on_batch_end(self, batch, logs=None):
        self.__batch += 1
        if self.__save_layer(self.__batch):
            out = self.__functor([self.__x_test, 0.])
            self.saved_layers.append(out)

    def get_saved_layers(self):
        return self.saved_layers


def get_parameters():
    parser = argparse.ArgumentParser()
    parameters_data(parser)
    network_parameters(parser)

    parser.add_argument('--bins', '-b',
                        dest='bins', default=30, type=int,
                        help="select the number of bins to use for bnning defaults to 30")

    parser.add_argument('--dest',
                        dest='dest', default="output",
                        help="destination folder for output files")

    parser.add_argument('--saved_epochs', '-se',
                        dest='no_saved_epochs', default=100, type=int,
                        help="no of epochs to consider when calculating mutual information")

    args = parser.parse_args()

    return args


def filename(args):
    name = "ts-" + "{0:.0%}".format(args.train_size) + ","
    name += "e-" + str(args.epochs) + ","
    name += "se-" + str(args.no_saved_epochs) + ","
    name += "_" + args.data_set + ","
    name += "_" + args.activation + ","
    name += "bs-" + str(args.batch_size) + ","
    name += "ns-" + str(args.shape)
    return name


print(__name__)
if __name__ == '__main__':
    main()
