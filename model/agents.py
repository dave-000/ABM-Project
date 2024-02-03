# Importing necessary libraries
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy

# Import functions from functions.py
from model.functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, floodplain_multipolygon


# Define the Households agent class
class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        #Sets richness of agent
        self.adaptive_capacity = model.wealth_distribution[unique_id]

        #Set expectation of autority
        self.expectation_authority = model.expectation_authority_distribution[unique_id]

        #Set Risk profile
        self.risk_profile = model.risk_profile_distribution[unique_id]

        #Set adaptation levels
        self.marginal_adaptation_level = 0
        self.total_adaptation_level = 0
        self.adaptiveDC = 0

        # getting flood map values
        # Get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = False
        if contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y):
            self.in_floodplain = True

        # Get the estimated flood depth at those coordinates.
        # the estimated flood depth is calculated based on the flood map (i.e., past data) so this is not the actual flood depth
        # Flood depth can be negative if the location is at a high elevation
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location, band=model.band_flood_img)
        # handle negative values of flood depth
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0

        # calculate the estimated flood damage given the estimated flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_estimated = calculate_basic_flood_damage(self, flood_depth=self.flood_depth_estimated)

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since there is not flood yet
        # and will update its value when there is a shock (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0

        #calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_actual = calculate_basic_flood_damage(self, flood_depth=self.flood_depth_actual)

    #Function to get friends who can be influencial
    def get_friends(self, radius):
        friends = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius)
        return friends

    # Function to count friends who can be influencial.
    def count_friends(self, radius):
        """Count the number of neighbors within a given radius (number of edges away). This is social relation and not spatial"""
        friends = self.get_friends(radius)
        return len(friends)

    def step(self):
        #NOTE modify with our processes
        # Logic for adaptation based on estimated flood damage and a random chance.
        # These conditions are examples and should be refined for real-world applications.
        self.adaptiveDC = (self.risk_profile + self.expectation_authority + self.flood_damage_estimated)/3
        increased_adaptation = self.adaptiveDC * self.adaptive_capacity

        if (self.total_adaptation_level + increased_adaptation) > 1:
            self.total_adaptation_level = 1
        else:
            self.total_adaptation_level += increased_adaptation
        self.adaptive_capacity -= 0.1 * self.adaptiveDC #Simulate the expense of resources caused by adaptation

        self.flood_damage_estimated = calculate_basic_flood_damage(self, flood_depth=(self.flood_depth_estimated - self.total_adaptation_level*6))



# Define the Government agent class
class Government(Agent):
    """
    A government agent that currently doesn't perform any actions.
    """
    def __init__(self, unique_id, model, gov_type):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model
        self.government_type = gov_type
        
        if gov_type == "democratic":
            self.gives_subsidies = True
            self.gives_PSA = True
        elif gov_type == "autocratic":
            self.gives_subsidies = False
            self.gives_PSA = False
        else:
            print("Unrecognised type of government")

    def step(self):
        #Subsidies
        if self.gives_subsidies is True:
            for i in range(self.model.number_of_households):
                if self.model.agents[i].adaptive_capacity < 0.2:
                    self.model.agents[i].adaptive_capacity += 0.1
        
        #PSA
        if self.gives_PSA is True:
            for i in range(self.model.number_of_households):
                self.model.agents[i].expectation_authority += 0.1
                self.model.agents[i].risk_profile += 0.1
                
                #Ensure no out of bound value
                if self.model.agents[i].expectation_authority > 1:
                    self.model.agents[i].expectation_authority = 1
                if self.model.agents[i].risk_profile > 1:
                    self.model.agents[i].risk_profile = 1

# More agent classes can be added here, e.g. for insurance agents.
