"""
Train % plot networks in the information plane
"""
from idnns.networks import information_network as inet
import time

def main():
    start = time.time()
    
    #Bulid the netowrk
    print ('*Building the network* (Step 1/4)')
    net = inet.informationNetwork()
    net.print_information()
    print ('*Start running the network* (Step 2/4)')
    net.run_network()
    print ('*Saving data* (Step 3/4)')
    net.save_data()
    print ('*Ploting figures* (Step 4/4)')
    #Plot the newtork
    net.plot_network()
    
    end = time.time()
    print("Time taken for main.py is " + end-start)
if __name__ == '__main__':
    main()

