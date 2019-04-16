"""
Train % plot networks in the information plane
"""
from idnns.networks import information_network as inet
import time


def main():
    start = time.time()

    # Build the network
    print('*Building the network with the following params:* (Step 1/4)')
    net = inet.informationNetwork()
    net.print_information()
    print('*Training and calculating the network information* (Step 2/4)')
    net.run_network()
    print('*Saving pickle to jobs directory* (Step 3/4)')
    net.save_data()
    print('*Plotting figures* (Step 4/4)')
    # Plot the network
    net.plot_network()

    end = time.time()
    print("Time taken for main.py is " + str(end - start))


if __name__ == '__main__':
    main()
