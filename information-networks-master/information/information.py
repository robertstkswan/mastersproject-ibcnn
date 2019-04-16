from information.NaftaliTishby import calculate_information_tishby
from information.kde import calculate_information_saxe

supported_estimators = ["Tishby", "KDE"]


def get_information_calculator(input_values, labels, entropy, bins):
    if entropy is None or entropy == "None":
        return lambda x: None
    elif entropy == "bins" or entropy == "Tishby":
        return calculate_information_tishby(input_values, labels, bins)
    elif entropy == "KDE":
        return calculate_information_saxe(input_values, labels, bins)
    else:
        raise ValueError("Unsupported mutual information estimator {}, available: {}"
                         .format(entropy, supported_estimators))


def mie_parameters(parser):
    parameters = parser.add_argument_group('Mutual Information Estimator parameters')

    parameters.add_argument('--mi_estimator',
                    '-mie', dest='mi_estimator', default="Tishby",
                    help="Choose what mutual information estimator to use available: {}, ".format(supported_estimators)
                         + "Tishby - method used by Tishby in his paper, "
                         "KDE - Kernel density estimator")

    parameters.add_argument('--bins',
                        '-b', dest='bins', default=-1, type=int,
                        help="select number of bins to use for MIE's. -1 for no binning. Note: Tishby MIE requires binning and defaults to 30")
