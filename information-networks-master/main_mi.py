import numpy as np
import argparse
import math
from networks.networks import network_parameters, get_model_categorical
from information.CalculateInformationCallback import CalculateInformationCallback
from information.information import get_information_calculator, mie_parameters
from data.data import load_data, parameters_data
from information.Processor import InformationProcessorDeltaApprox, InformationProcessorUnion
from information.Processor import information_processor_parameters
from utils import ProgressBarCallback


def main():
    args = get_parameters()

    (x_train, y_train), (x_test, y_test), categories = load_data(args.data_set, args.train_size)

    x_full = np.concatenate((x_train, x_test))
    y_full = np.concatenate((y_train, y_test))

    if ',' not in args.mi_estimator:
        information_calculator = get_information_calculator(x_full, y_full, args.mi_estimator, args.bins)
        processor = InformationProcessorDeltaApprox(information_calculator)
    else:
        mies = args.mi_estimator.split(',')
        calculators = [get_information_calculator(x_full, y_full, mie, args.bins) for mie in mies]
        ips = [InformationProcessorDeltaApprox(calc) for calc in calculators]
        processor = InformationProcessorUnion(ips)

    model = get_model_categorical(
        input_shape=x_train[0].shape,
        network_shape=args.shape,
        categories=categories,
        activation=args.activation)

    print("Training and Calculating mutual information")
    batch_size = min(args.batch_size, len(x_train)) if args.batch_size > 0 else len(x_train)
    no_of_batches = math.ceil(len(x_train) / batch_size) * args.epochs
    information_callback = CalculateInformationCallback(model, processor, x_full)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=[information_callback, ProgressBarCallback(no_of_batches)],
              epochs=args.epochs,
              validation_data=(x_test, y_test),
              verbose=0)

    append = ",b-" + str(information_callback.batch)
    print("Saving data to files")
    processor.save(args.dest + "/data/" + filename(args) + append)
    print("Producing image")
    processor.plot(args.dest + "/images/" + filename(args) + append)
    print("Done")
    return


def get_parameters():
    parser = argparse.ArgumentParser()
    parameters_data(parser)
    network_parameters(parser)
    mie_parameters(parser)
    information_processor_parameters(parser)

    parser.add_argument('--dest',
                        dest='dest', default="output",
                        help="destination folder for output files")

    args = parser.parse_args()

    return args


def filename(args):
    name = "ts-" + "{0:.0%}".format(args.train_size) + ","
    name += "e-" + str(args.epochs) + ","
    name += "_" + args.activation
    name += "_" + args.data_set + ","
    name += "mie-" + str(args.mi_estimator) + ","
    name += "bs-" + str(args.batch_size) + ","
    if args.bins != 1:
        name += "bins-" + str(args.bins) + ","
    name += "ns-" + str(args.shape)
    return name


if __name__ == '__main__':
    main()
