# Importing necessary libraries
import networkx as nx
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import geopandas as gpd
import rasterio as rs
import matplotlib.pyplot as plt
import numpy as np

# Import the agent class(es) from agents.py
from model.agents import Households, Government

# Import functions from functions.py
from model.functions import get_flood_map_data, calculate_basic_flood_damage, calculate_influenced_risk_profile
from model.functions import map_domain_gdf, floodplain_gdf


# Define the AdaptationModel class
class AdaptationModel(Model):
    """
    The main model running the simulation. It sets up the network of household agents,
    simulates their behavior, and collects data. The network type can be adjusted based on study requirements.
    """

    def __init__(self,
                 seed = 1,
                 number_of_households = 25, # number of household agents
                 # Simplified argument for choosing flood map. Can currently be "harvey", "100yr", or "500yr".
                 flood_map_choice='harvey',
                 # ### network related parameters ###
                 # The social network structure that is used.
                 # Can currently be "erdos_renyi", "barabasi_albert", "watts_strogatz", or "no_network"
                 network = 'watts_strogatz',
                 # likeliness of edge being created between two nodes
                 probability_of_network_connection = 0.4,
                 # number of edges for BA network
                 number_of_edges = 3,
                 # number of nearest neighbours for WS social network
                 number_of_nearest_neighbours = 5,
                 # government type
                 government_type = "democratic"
                 ):

        super().__init__(seed = seed)

        # defining the variables and setting the values
        self.number_of_households = number_of_households  # Total number of household agents
        self.seed = seed
        np.random.seed(self.seed)

        # network
        self.network = network # Type of network to be created
        self.probability_of_network_connection = probability_of_network_connection
        self.number_of_edges = number_of_edges
        self.number_of_nearest_neighbours = number_of_nearest_neighbours

        # generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network()

        # create grid out of network graph
        self.grid = NetworkGrid(self.G)

        #Initialize wealth
        self.initialize_wealth()

        #Initialize expectation to authority
        self.initialize_expectation_authority()

        #Initialize risk profile
        self.initialize_risk_profile()
        self.next_risk_profile = np.zeros(self.number_of_households)

        # Initialize maps
        self.initialize_maps(flood_map_choice)

        # set schedule for agents
        self.schedule = RandomActivation(self)  # Schedule for activating agents

        # create households through initiating a household on each node of the network graph
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self)
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)
        
        # create government through initiating one Government object
        government = Government(unique_id = 0, model = self, gov_type = government_type)
        self.schedule.add(government)

        # Builds the trust matrix
        self.trust_matrix = np.random.normal(0.5, 0.4, size = (number_of_households, number_of_households))
        for i in range(number_of_households):
            for j in range(number_of_households):
                if self.trust_matrix[i,j] > 1:
                    self.trust_matrix[i,j] = 1
                if self.trust_matrix[i,j] < -1:
                    self.trust_matrix[i,j] = -1

        # You might want to create other agents here, e.g. insurance agents.

        # Data collection setup to collect data
        model_metrics = {
                        "total_adapted_households": self.total_adapted_households,
                        # ... other reporters ...
                        }

        agent_metrics = {
                        "FloodDepthEstimated": "flood_depth_estimated",
                        "FloodDamageEstimated" : "flood_damage_estimated",
                        "FloodDepthActual": "flood_depth_actual",
                        "FloodDamageActual" : "flood_damage_actual",
                        "IsAdapted": "is_adapted",
                        #"FriendsCount": lambda a: a.count_friends(radius=1),
                        "location":"location",
                        # ... other reporters ...
                        }
        #set up the data collector
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)


    def initialize_network(self):
        """
        Initialize and return the social network graph based on the provided network type using pattern matching.
        """
        if self.network == 'erdos_renyi':
            return nx.erdos_renyi_graph(n=self.number_of_households,
                                        p=self.number_of_nearest_neighbours / self.number_of_households,
                                        seed=self.seed)
        elif self.network == 'barabasi_albert':
            return nx.barabasi_albert_graph(n=self.number_of_households,
                                            m=self.number_of_edges,
                                            seed=self.seed)
        elif self.network == 'watts_strogatz':
            return nx.watts_strogatz_graph(n=self.number_of_households,
                                        k=self.number_of_nearest_neighbours,
                                        p=self.probability_of_network_connection,
                                        seed=self.seed)
        elif self.network == 'no_network':
            G = nx.Graph()
            G.add_nodes_from(range(self.number_of_households))
            return G
        else:
            raise ValueError(f"Unknown network type: '{self.network}'. "
                            f"Currently implemented network types are: "
                            f"'erdos_renyi', 'barabasi_albert', 'watts_strogatz', and 'no_network'")

    def initialize_wealth(self):
        distribution_vector = np.random.randint(100, size=(self.number_of_households))
        wealth_vector = []
        for i in range(self.number_of_households):
            wealth_value = np.random.normal(1, 0.05)
            if distribution_vector[i] < 71:
                wealth_value = np.random.normal(0.75, 0.05)
            if distribution_vector[i] < 58:
                wealth_value = np.random.normal(0.5, 0.05)
            if distribution_vector[i] <40:
                wealth_value = np.random.normal(0.25, 0.05)
            if distribution_vector[i] < 22:
                wealth_value = np.random.normal(0, 0.05)

            if wealth_value < 0:
                wealth_value = 0
            if wealth_value > 1:
                wealth_value = 1
            wealth_vector.append(wealth_value)
        self.wealth_distribution = wealth_vector


    def initialize_expectation_authority(self):
        distribution_vector = np.random.randint(100, size=(self.number_of_households))
        expectation_vector = []
        for i in range(self.number_of_households):
            expectation_value = np.random.normal(1, 0.05)
            if distribution_vector[i] < 88:
                expectation_value = np.random.normal(0.75, 0.05)
            if distribution_vector[i] < 66:
                expectation_value = np.random.normal(0.5, 0.05)
            if distribution_vector[i] < 43:
                expectation_value = np.random.normal(0.25, 0.05)
            if distribution_vector[i] < 30:
                expectation_value = np.random.normal(0, 0.05)

            if expectation_value < 0:
                expectation_value = 0
            if expectation_value > 1:
                expectation_value = 1
            expectation_vector.append(expectation_value)
        self.expectation_authority_distribution = expectation_vector


    def initialize_risk_profile(self):
        distribution_vector = np.random.randint(100, size=(self.number_of_households))
        risk_profile_vector = []
        for i in range(self.number_of_households):
            risk_value = np.random.normal(0.85, 0.05)
            if distribution_vector[i] < 87:
                risk_value = np.random.normal(0.725, 0.05)
            if distribution_vector[i] < 84:
                risk_value = np.random.normal(0.6, 0.05)
            if distribution_vector[i] < 45:
                risk_value = np.random.normal(0.3, 0.05)

            if np.random.rand() < 0.3:
                risk_value += np.random.normal(0.4, 0.05)
            if risk_value < 0:
                risk_value = 0
            if risk_value > 1:
                risk_value = 1
            risk_profile_vector.append(risk_value)
        self.risk_profile_distribution = risk_profile_vector


    def initialize_maps(self, flood_map_choice):
        """
        Initialize and set up the flood map related data based on the provided flood map choice.
        """
        # Define paths to flood maps
        flood_map_paths = {
            'harvey': "input_data/floodmaps/Harvey_depth_meters.tif",
            '100yr': "input_data/floodmaps/100yr_storm_depth_meters.tif",
            '500yr': "input_data/floodmaps/500yr_storm_depth_meters.tif"  # Example path for 500yr flood map
        }

        # Throw a ValueError if the flood map choice is not in the dictionary
        if flood_map_choice not in flood_map_paths.keys():
            raise ValueError(f"Unknown flood map choice: '{flood_map_choice}'. "
                             f"Currently implemented choices are: {list(flood_map_paths.keys())}")

        # Choose the appropriate flood map based on the input choice
        flood_map_path = flood_map_paths[flood_map_choice]

        # Loading and setting up the flood map
        self.flood_map = rs.open(flood_map_path)
        self.band_flood_img, self.bound_left, self.bound_right, self.bound_top, self.bound_bottom = get_flood_map_data(
            self.flood_map)

    def total_adapted_households(self):
        #NOTE Change considering that the adaptation is not 0 or 1
        """Return the total number of households that have adapted."""
        #BE CAREFUL THAT YOU MAY HAVE DIFFERENT AGENT TYPES SO YOU NEED TO FIRST CHECK IF THE AGENT IS ACTUALLY A HOUSEHOLD AGENT USING "ISINSTANCE"
        adapted_count = sum([1 for agent in self.schedule.agents if isinstance(agent, Households)])
        return adapted_count

    def plot_model_domain_with_agents(self):
        fig, ax = plt.subplots()
        # Plot the model domain
        map_domain_gdf.plot(ax=ax, color='lightgrey')
        # Plot the floodplain
        floodplain_gdf.plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)

        # Collect agent locations and statuses
        for agent in self.schedule.agents:
            color = 'blue' if agent.is_adapted else 'red'
            ax.scatter(agent.location.x, agent.location.y, color=color, s=10, label=color.capitalize() if not ax.collections else "")
            ax.annotate(str(agent.unique_id), (agent.location.x, agent.location.y), textcoords="offset points", xytext=(0,1), ha='center', fontsize=9)
        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Red: not adapted, Blue: adapted")

        # Customize plot with titles and labels
        plt.title(f'Model Domain with Agents at Step {self.schedule.steps}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def step(self):
        """
        introducing a shock: 
        at time step 5, there will be a global flooding.
        This will result in actual flood depth. Here, we assume it is a random number
        between 0.5 and 1.2 of the estimated flood depth. In your model, you can replace this
        with a more sound procedure (e.g., you can devide the floop map into zones and 
        assume local flooding instead of global flooding). The actual flood depth can be 
        estimated differently
        """

        #Calculates the updated risk profile (NOTE, the risk profile are updated with the new values only at the END of the step)
        self.next_risk_profile = calculate_influenced_risk_profile(self)

        if self.schedule.steps == 5:
            for agent in self.schedule.agents:
                # Calculate the actual flood depth as a random number between 0.5 and 1.2 times the estimated flood depth
                agent.flood_depth_actual = np.random.uniform(0.5, 1.2) * agent.flood_depth_estimated
                # calculate the actual flood damage given the actual flood depth
                agent.flood_damage_actual = calculate_basic_flood_damage(agent, agent.flood_depth_actual)

        # Collect data and advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()

        for i in range(self.number_of_households):
            self.agents[i].risk_profile = self.next_risk_profile[i]
