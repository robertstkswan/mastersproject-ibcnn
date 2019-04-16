from plot.plot import plot_movie
import argparse
import _pickle


def main():
    args = movie_parameters()
    # data -> epoch * (i_x_t, i_y_t, i_t_t)
    data = _pickle.load(open(args.input, 'rb'))
    plot_movie(data, args.movie_length, filename=args.output)


def movie_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        help='input pickle data file')

    parser.add_argument('output',
                        help='output movie file')

    parser.add_argument('--movie_length',
                        '-ml', dest='movie_length', default=40, type=int,
                        help='approx movie length in seconds')

    args = parser.parse_args()
    args.output = output(args)
    return args


def output(args):
    if args.output is not None:
        return args.output
    out = args.input.split('/')[-1].split('.')[0]
    out = out[:-7] if out.endswith('_pickle') else out
    return "output/movies/" + out


if __name__ == '__main__':
    main()

