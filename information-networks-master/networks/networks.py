from keras.layers import Dropout, Dense, BatchNormalization
from keras import Sequential

activation_functions = ["tanh, sigmoid, relu, linear"]


def get_model_categorical(input_shape, network_shape="10,8,6,4", categories=2, activation='tanh'):
    model = Sequential()
    network_shape = network_shape.split(',')

    model.add(Dense(input_shape[0], activation=activation, input_shape=input_shape))

    for layer_spec in network_shape:
        model.add(decode_layer(layer_spec, activation))

    model.add(Dense(categories, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def decode_layer(spec, activation="tanh"):
    if spec == "Dropout" or spec == "Dr":
        spec = spec.split('-')
        dropout_rate = float(spec[1]) if len(spec) >= 2 else 0.2
        return Dropout(rate=dropout_rate)
    elif spec == "BatchNormalization" or spec == "BN":
        return BatchNormalization()
    else:
        spec = spec.split('-')
        size = int(spec[0])
        activation = spec[1] if len(spec) >= 2 else activation
        return Dense(size, activation=activation)


def network_parameters(parser):
    parameters = parser.add_argument_group('Neural Network parameters')

    parameters.add_argument('--activation_function',
                        '-af', dest='activation', default="tanh",
                        help="Choose what neural network activation function to use available: {}"
                        .format(activation_functions))

    parameters.add_argument('--network_shape', '-ns', dest='shape', default="10,8,6,4",
                        help='Shape of the DNN, ex :'
                             '12,Dr,10-tanh,8-relu,6-sigmoid,BN,2 , would represent a DNN shape where 1st layer is Dense of size 12, 2nd layer is a Dropout layer, 3rd layer is Dense with size 10 and tanh activation function, 5th is Dense with relu activation function,..., 7th is BatchNormalization layer,..., note: 0th and last layers are automatically created to fit the dataset')

    parameters.add_argument('--batch_size',
                        '-bs', dest='batch_size', default=512,
                        type=int)

    parameters.add_argument('--num_of_epochs',
                        '-e', dest='epochs', default=1500,
                        type=int, help='Number of times to scan the dataset for NN training')

