import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import cv2 as cv
import av
import io
import random
import os
from itertools import repeat
from matplotlib.backends.backend_agg import FigureCanvas

class SIR:
    def __init__(self, network_type, time = 1000, population = 1000, num_infect = 50, num_removed = 0, net_size = 4, rate_si = 0.05, rate_ir = 0.01, folder = "simulation", **kwargs):
        """
        Initialize the model at time 0
        
        Params:
            network_type (str): the type of network
            time (int): total time steps
            population (int): # of individuals in our network
            net_size (int): # of individuals in one personal network
            infectious (int): # of infectious (I) individuals at time 0
            removed (int): # of immuned (R) individuals at time 0
            rate_si (float): base rate 'beta' from S to I = 1 - immune_rate
            rate_ir (float): base rate 'gamma' from I to R
        """
        # Initialize params
        self.folder = folder
        self.time, self.population = time, population,
        self.net_size, self.net_type = net_size, network_type
        self.connect_rate = net_size / population
        self.infectious, self.removed = num_infect, num_removed
        self.suscept = population - num_infect - num_removed
        self.rate_si, self.rate_ir = rate_si, rate_ir
        self.rate_reproduction = net_size * rate_si
        
        # Set the randomization and get kwargs
        np.random.seed(242)
        quarantine = kwargs.get('is_quarantine', False)
        social_distancing = kwargs.get('is_social_distancing', False)
        online = kwargs.get('is_online', False)
        
        # Simulation params
        self.network, self.status = self.init_network(network_type), self.init_status()
        self.results = None
        
        # Run the model
        self.covid_simulation = self.run(quarantine, social_distancing, online)
    
    
    def init_network(self, n_type):
        """
        Create a Networkx Graph representing a network of people based on the network's type.
        The network is symmetric.
        
        Available type:
            "random": Erdos-Renyi network
            "scale-free": Scale-free network
            "small-world": Watts-Strogatz network
        """
        # Initialize network
        if n_type == "scale-free":  
            network = nx.scale_free_graph(self.population)
            network = network.to_undirected()
        elif n_type == "small-world":
            REWIRING_RATE = 0.5
            network = nx.connected_watts_strogatz_graph(self.population, self.net_size, REWIRING_RATE)
        else:
            network = nx.erdos_renyi_graph(self.population, self.connect_rate)
                
        return network
    
    
    def init_status(self):
        """
        Pick the first `num_infect` people to be infected with Covid-19.
        The rest are susceptible. No removed and quarantine.
        
        Status:
            0: susceptible
            1: infectious
            2: quarantine
            3: removed
        """
        # Initialize status
        statuses = np.zeros(self.population)
        for person in range(self.infectious):
            statuses[person] = 1
                
        return statuses
    
    
    def update_status(self, quarantine, social_distancing):    
        """
        Update the network's status
        """
        # Get current simuation params
        network, status = self.network.copy(), self.status.copy()
        
        # Update infectious
        infected = np.where(status == 1)[0]
        for person in infected:
            # Apply social distancing
            social_rate = 0.5 if social_distancing else 1
            for friend in network[person]:
                if np.random.uniform(0,1) < self.rate_si and np.random.uniform(0,1) <= social_rate:
                    status[friend] = 1
                
            # Be moved to quarantine if set `quarantine = True`
            status[person] = 2 if quarantine else 1
                
        
        # Update removed
        has_covid = np.where((status == 1) | (status == 2))[0]
        for person in has_covid:
            if np.random.uniform(0,1) < self.rate_ir:
                status[person] = 3
                    
        # Update simuation params
        self.network, self.status = network, status
        susceptible, infectious, quarantine, removed = (len(np.where(status == i)[0]) for i in [0, 1, 2, 3])
        self.suscept, self.infectious, self.removed = susceptible, infectious + quarantine, removed
        
        return network, status, susceptible, infectious + quarantine, removed

        
    def run(self, quarantine, social_distancing, online):
        """
        Run the model
        """
        # Initialize list to store data at all time steps
        suscept, infectious, removed = [self.suscept] * 2, [self.infectious] * 2, [self.removed] * 2
        all_network = []

        # Recursively run
        for step in range(2, self.time + 3):
            # Store data at all time steps
            self.results = pd.DataFrame.from_dict({'Time': list(range(step)), 'Susceptible': suscept, 'Infectious': infectious, 'Removed': removed}, orient = 'index').transpose()
            
            # Visualize network
            curr_network = self.make_plot(filename = str(step))
            all_network.append(curr_network)
            
            # Update network
            _, _, this_suscept, this_infectious, this_removed = self.update_status(quarantine, social_distancing)
            suscept.append(this_suscept)
            infectious.append(this_infectious)
            removed.append(this_removed)
            
            # Update reproudction rate
            self.rate_reproduction = (infectious[-1] - infectious[-2]) / infectious[-2]

        return all_network
            
            
    def visualize_network(self, fig):
        """
        Visualize the network with their member's Covid status. 
        """
        # Make sub-figure
        ax = fig.add_subplot(1, 2, 2)
        
        # Get current simuation params
        network, status = self.network.copy(), self.status.copy()
        colors = ["green" if i == 3 else "gray" if i == 2 else "red" if i == 1 else "blue" for i in status]
          
        # Draw diagram
        pos = nx.spring_layout(network)
        nx.draw_networkx_nodes(network, pos, node_size = 100, node_color = colors, ax = ax)
        ax.set_title("Network")
    
    
    def visualize_statuses(self, fig):
        """
        Visualize the Covid statuses of all members in the network
        """
        # Make sub-figure
        ax = fig.add_subplot(1, 2, 1)
        
        # Get current simuation params
        time = self.results['Time']
        percent_suscept = self.results['Susceptible'] / self.population * 100
        percent_infect = self.results['Infectious'] / self.population * 100
        percent_removed = self.results['Removed'] / self.population * 100
        
        # Extend to cover full period
        new_time = self.time + 2
        new_period = range(new_time)
        percent_suscept = pd.concat([percent_suscept, pd.Series([0] * (new_time - len(time)))])
        percent_infect = pd.concat([percent_infect, pd.Series([0] * (new_time - len(time)))])
        percent_removed = pd.concat([percent_removed, pd.Series([0] * (new_time - len(time)))])
        
        # Plot SIR
        ax.stackplot(new_period, percent_suscept, percent_infect, percent_removed, colors = ['blue', 'red', 'green'], step = 'pre')
        ax.set_xlabel('Time')
        ax.set_ylabel('% Population')
        ax.set_title("Covid statuses")
        ax.legend(['Susceptible', 'Infectious', 'Removed'], prop = {'size': 10}, loc = 'upper center', bbox_to_anchor = (0.5, 1.02), ncol = 3)
        
        
    def make_histogram(self, filename = "degree"):
        """
        Visualize the network's degree histogram
        """
        # Get network degree
        network = self.network.copy()
        degree_sequence = sorted((d for n, d in network.degree()), reverse = True)
        
        # Draw diagram
        fig = plt.figure(figsize = (16, 9))
        plt.bar(*np.unique(degree_sequence, return_counts = True))
        plt.title(f"Degree histogram of {self.net_type}")
        plt.xlabel("Degree")
        plt.ylabel("# of Nodes")
        
        # Save diagrams
        plt.savefig(f"outputs/{self.net_type}/{filename}.jpg")
        plt.close()

    
    def make_plot(self, filename = "test"):
        """
        Visualize network and its Covid status
        """
        # Get current simuation params
        network, status = self.network.copy(), self.status.copy()
        
        # Draw diagram
        fig = plt.figure(figsize = (16, 9))
        
        # Draw network
        self.visualize_network(fig)
        
        # Draw status
        self.visualize_statuses(fig)
        
        # Add legend and title
        fig.suptitle(r'{0} network, $\beta = {1}, \gamma = {2}, R_0 = {3}$'.format(self.net_type, self.rate_si, self.rate_ir, round(self.rate_reproduction, 3)))
        
        # Save diagrams
        plt.savefig(f"outputs/{self.net_type}/{self.folder}/{filename}.jpg")
        plt.close()

        
    def make_video(self, name = 'test.mp4'):
        """
        Make video from sequential frames
        """
        os.system(f"ffmpeg -f image2 -r 5 -i outputs/{self.net_type}/{self.folder}/%01d.jpg -vcodec mpeg4 -y ./outputs/{self.net_type}/{name}")


class SIROnline(SIR):
    def make_histogram(self, filename = "degree"):
        """
        Visualize the network's degree histogram
        """
        # Get network degree
        network = self.network.copy()
        degree_sequence = sorted((d for n, d in network.degree()), reverse = True)
        
        # Draw diagram
        fig = plt.figure(figsize = (16, 9))
        plt.bar(*np.unique(degree_sequence, return_counts = True))
        plt.title(f"Degree histogram of {self.net_type}")
        plt.xlabel("Degree")
        plt.ylabel("# of Nodes")
        
        # Convert figure to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.array(fig.canvas.get_renderer()._renderer)
        plt.close()
        
        return img
        
    def make_plot(self, filename = "test"):
        """
        Visualize network and its Covid status
        """
        # Get current simuation params
        network, status = self.network.copy(), self.status.copy()
        
        # Draw diagram
        fig = plt.figure(figsize = (16, 9))
        
        # Draw network
        self.visualize_network(fig)
        
        # Draw status
        self.visualize_statuses(fig)
        
        # Add legend and title
        fig.suptitle(r'{0} network, $\beta = {1}, \gamma = {2}, R_0 = {3}$'.format(self.net_type, self.rate_si, self.rate_ir, round(self.rate_reproduction, 3)))
        
        # Convert figure to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = cv.cvtColor(np.array(fig.canvas.get_renderer()._renderer), cv.COLOR_RGB2BGR)
        plt.close()
        
        return img
    
    def make_video(self, name = 'test.mp4'):
        """
        Make video on request from Streamlit
        """
        # Select video resolution and frame rate.
        frames = self.covid_simulation
        n_frames, width, height, fps = len(frames), frames[0].shape[1], frames[0].shape[0], 5
        video = io.BytesIO() 

         # Open "in memory file" as MP4 video output
        output = av.open(video, 'w', format = "mp4")
        stream = output.add_stream('h264', str(fps))
        stream.width = width; stream.height = height
        stream.pix_fmt = 'yuv420p'; stream.options = {'crf': '17'}

        # Iterate the created images, encode and write to MP4 memory file.
        for i in range(n_frames):
            img = frames[i]
            frame = av.VideoFrame.from_ndarray(img, format = 'bgr24') 
            packet = stream.encode(frame)  # Encode video frame
            output.mux(packet)  # "Mux" the encoded frame (add the encoded frame to MP4 file).

        # Flush the encoder
        packet = stream.encode(None)
        output.mux(packet)
        output.close()

        # Seek to the beginning of the BytesIO.
        video.seek(0)  
        return video

        
#
#def main():
#    # Set params
#    networks = ["random", "scale-free", "small-world"]
#    POPULATION = 300000
#    TIME = 30 # days
#    RATE_IMMUNE = 0.53
#    RATE_SI = 1 - RATE_IMMUNE
#    RATE_RI = 0.178
#
#    # Random network without quarantine
#    rd = SIR(networks[0], time = TIME, rate_si = RATE_SI, rate_ir = RATE_RI)
#    rd.make_histogram()
#    rd.make_video("simulation.mp4")
#
#    # Small world network without quarantine
#    sw = SIR(networks[2], time = TIME, rate_si = RATE_SI, rate_ir = RATE_RI)
#    sw.make_histogram()
#    sw.make_video("simulation.mp4")
#
#    # Scale-free network without quarantine
#    sf1 = SIR(networks[1], time = TIME, rate_si = RATE_SI, rate_ir = RATE_RI)
#    sf1.make_histogram()
#    sf1.make_video("simulation.mp4")
#
#    # Scale-free network with quarantine
#    sf2 = SIR(networks[1], time = TIME, rate_si = RATE_SI, rate_ir = RATE_RI, folder = "quarantine", is_quarantine = True)
#    sf2.make_video("quarantine.mp4")
#
#    # Scale-free network with social distancing
#    sf3 = SIR(networks[1], time = TIME, rate_si = RATE_SI, rate_ir = RATE_RI, folder = "social_distancing", is_social_distancing = True)
#    sf3.make_video("social_distancing.mp4")
#
#    # Scale-free network with quarantine and social distancing
#    sf4 = SIR(networks[1], time = TIME, rate_si = RATE_SI, rate_ir = RATE_RI, folder = "pandemic", is_quarantine = True, is_social_distancing = True)
#    sf4.make_video("pandemic.mp4")
#
#if __name__ == "__main__":
#    main()
