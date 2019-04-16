import _pickle
import abc
from plot.plot import plot_main
from BlockingThreadPoolExecutor import BlockingThreadPoolExecutor
from threading import Lock


class InformationProcessor(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def save(self, path):
        """
        Saves the data to a pickle file
        :param path: file path
        :return: None
        """
        raise NotImplementedError(".save() is not implemented (are you using the Abstract class?)")

    @abc.abstractmethod
    def plot(self, path, show=False):
        """
        Plots the data and saves it to a file
        :param path: file path
        :param show: show image?
        :return: None
        """
        raise NotImplementedError(".plot() is not implemented (are you using the Abstract class?)")

    @abc.abstractmethod
    def calculate_information(self, activation, epoch):
        """
        Might calculate mutual information for the epoch
        :param activation: function returning network activation data, see @Information.CalculateInformationCallback
                           for how the data is gathered
        :param epoch: epoch number
        :return: None
        """
        raise NotImplementedError(".calculate_information() is not implemented (are you using the Abstract class?)")

    def finish_information_calculation(self):
        raise NotImplementedError(
            ".finish_information_calculation() is not implemented (are you using the Abstract class?)")


class InformationProcessorSimple(InformationProcessor):

    def __init__(self, information_calculator, skip):
        """
        :param information_calculator: See information.information.get_information_calculator
        :param skip: if skip is int: calculate MI for: epoch % skip == 0
                     if skip is a function int -> bool: calculate MI for: skip(epoch) == true
        """
        self.mi = {}
        self.__prev = None
        if type(skip) is int:
            self.__skip = lambda x: x % skip == 0
        else:
            self.__skip = skip
        self.__calculator = information_calculator

    def save(self, path):
        _pickle.dump(self.mi, open(path, 'wb'))

    def plot(self, path, show=False):
        # this line just unpacks the map so it's easy to feed into plot_main
        mi = list(zip(*map(lambda el: (el[0], *el[1]), self.mi.items())))
        epochs, i_x_t, i_y_t, i_t_t = mi
        plot_main(i_x_t, i_y_t, path, show)

    def calculate_information(self, activation, epoch):
        if self.__skip(epoch):
            mi = self.__calculator(activation())
            self.mi[epoch] = mi

    def finish_information_calculation(self):
        # there is no state to cleanup for this class
        pass


class InformationProcessorDeltaApprox(InformationProcessor):
    """
    Contains the logic for which epochs to calculate Mutual Information

    Distance between consecutive calculated epochs is Approximately @delta
    """

    def __init__(self, information_calculator, delta=0.2):
        """
        :param information_calculator: See information.information.get_information_calculator
        :param delta: how densely to calculate information, higher val => more layers skipped
        """
        self.mi = {}
        self.__prev = None
        self.__skip = 1
        self.__delta = delta
        self.__calculator = information_calculator

    def save(self, path):
        _pickle.dump(self.mi, open(path, 'wb'))

    def plot(self, path, show=False):
        # this line just unpacks the map so it's easy to feed into plot_main
        mi = list(zip(*map(lambda el: (el[0], *el[1]), self.mi.items())))
        epochs, i_x_t, i_y_t, i_t_t = mi
        plot_main(i_x_t, i_y_t, path, show)

    def calculate_information(self, activation, epoch):
        if epoch % self.__skip != 0:
            return

        activation = activation()

        if self.__prev is None:
            self.__prev = self.__calculator(activation)
            self.mi[epoch] = self.__prev
            return

        curr = self.__calculator(activation)
        self.mi[epoch] = curr
        if _dist(self.__prev, curr) <= self.__delta:
            self.__skip = self.__skip * 2

    def finish_information_calculation(self):
        # there is no state to cleanup for this class
        pass


class InformationProcessorDeltaExact(InformationProcessor):
    """
    Contains the logic for which epochs to calculate Mutual Information

    (*) Distance between consecutive calculated epochs is guaranteed to be less than @delta

    Uses a lot of memory hence @buffer_limit

    In case of OOMs use InformationProcessorDeltaApprox

    The code is quite complicated but is stable, use this only if you need the guarantee (*), o/w use *DeltaApprox
    """
    def __init__(self, information_calculator, buffer_limit=1, delta=0.2, max_workers=1):
        """
        :param information_calculator: See information.information.get_information_calculator
        :param buffer_limit: limits memory usage, higher val => more memory usage (bad), more layers skipped (good)
        :param delta: how densely to calculate information, higher val => more layers skipped
        :param max_workers: how many threads to use, (note: information_calculator may also be parallelized in which
                            case max_workers=1 may be a good choice)
        """
        self.mi = {}
        self.__global_prev = None
        self.__buffered_activations = []
        self.__buffer_limit = buffer_limit
        self.__delta = delta
        self.__lock = Lock()
        self.__calculator = information_calculator
        self.__executor = BlockingThreadPoolExecutor(max_workers=max_workers)

    def save(self, path):
        _pickle.dump(self.mi, open(path, 'wb'))

    def plot(self, path, show=False):
        # this line just unpacks the map so it's easy to feed into plot_main
        mi = list(zip(*map(lambda el: (el[0], *el[1]), self.mi.items())))
        epochs, i_x_t, i_y_t, i_t_t = mi
        plot_main(i_x_t, i_y_t, path, show)

    def calculate_information(self, activation, epoch):
        activation = activation()
        self.__lock.acquire()
        if self.__global_prev is None:
            self.__global_prev = self.__calculator(activation)
            self.mi[epoch] = self.__global_prev
            self.__lock.release()
            return

        self.__buffered_activations.append((activation, epoch))
        if len(self.__buffered_activations) >= self.__buffer_limit:
            # copy and clear __buffered_activations
            activation_buffer = self.__buffered_activations
            self.__buffered_activations = []

            local_prev = self.__global_prev

            # pre-compute next global_prev
            curr_activation, epoch_curr = activation_buffer[-1]

            mi_curr = self.__calculator(curr_activation)
            self.__global_prev = mi_curr
            if _dist(local_prev, mi_curr) <= self.__delta:
                self.__buffer_limit = min(self.__buffer_limit * 2, 32)
            self.__lock.release()
            self.__executor.submit(self.__info_calc_entry, local_prev, mi_curr, epoch_curr, activation_buffer, [])
            return
        self.__lock.release()

    def finish_information_calculation(self):
        if len(self.__buffered_activations) > 0:
            self.__executor.submit(self.__info_calc_entry)
        self.__executor.shutdown()

    def __info_calc_entry(self, local_prev, mi_curr, epoch_curr, activation_buffer, carry):
        self._info_calc_inner_loop(local_prev, mi_curr, epoch_curr, activation_buffer, carry)
        with self.__lock:
            for epoch, mi in carry:
                self.mi[epoch] = mi

    def __info_calc_loop(self, mi_prev, activation_buffer, carry):
        assert (len(activation_buffer) > 0)
        curr_activation, epoch_curr = activation_buffer[-1]
        mi_curr = self.__calculator(curr_activation)

        return self._info_calc_inner_loop(mi_prev, mi_curr, epoch_curr, activation_buffer, carry)

    def _info_calc_inner_loop(self, mi_prev, mi_curr, epoch_curr, activation_buffer, carry):
        carry.append((epoch_curr, mi_curr))
        while _dist(mi_prev, mi_curr) > self.__delta:
            split = int(len(activation_buffer) / 2)
            if split == 0:
                break  # _dist(i, i+1) > delta, no further division is possible
            mi_prev = self.__info_calc_loop(mi_prev, activation_buffer[:split], carry)
            activation_buffer = activation_buffer[split:]
        return mi_curr


class InformationProcessorUnion(InformationProcessor):
    def __init__(self, ips):
        assert (len(ips) > 0)
        super().__init__(None)
        self.ips = ips

    def calculate_information(self, activation, epoch):
        for ip in self.ips:
            ip.calculate_information(activation, epoch)

    def finish_information_calculation(self):
        for ip in self.ips:
            ip.finish_information_calculation()

    def save(self, path):
        for (i, ip) in enumerate(self.ips):
            ip.save(path=path + "_{}".format(i))

    def plot(self, path, show=False):
        for (i, ip) in enumerate(self.ips):
            ip.plot(path=path + "_{}".format(i), show=show)


def information_processor_parameters(parser):
    parameters = parser.add_argument_group('Information Processor parameters')

    parameters.add_argument('--delta',
                            '-d', dest='delta', default=0.1,
                            type=float,
                            help="Tolerance on how densely to calculate mutual information, higher delta will skip more epochs")

    parameters.add_argument('--cores',
                            '-c', dest='cores', default=1,
                            type=int,
                            help='number of information instances to compute at a time')


def _dist(i_a, i_b):
    """
    Just a random distance metric used to decide if to compute mutual
    information for nearby epochs

    :param i_a: information for epoch a 
    :param i_b: information for epoch b
    :return: some notion of distance 
    """
    d = max(
        max(abs(i_a[0] - i_b[0])),
        max(abs(i_a[1] - i_b[1])),
        max(abs(i_a[2] - i_b[2])),
    )
    return d
