# -*- coding: utf-8 -*-
"""
@author: thoridwagenblast

Functions that are used in the model_file.py and agent.py for the running of the Flood Adaptation Model.
Functions get called by the Model and Agent class.
"""
import numpy as np
import math
from shapely import contains_xy
from shapely import prepare
import geopandas as gpd
import matplotlib.pyplot as plt
import os

def set_initial_values(input_data, parameter, seed):
    """
    Function to set the values based on the distribution shown in the input data for each parameter.
    The input data contains which percentage of households has a certain initial value.
    
    Parameters
    ----------
    input_data: the dataframe containing the distribution of paramters
    parameter: parameter name that is to be set
    seed: agent's seed
    
    Returns
    -------
    parameter_set: the value that is set for a certain agent for the specified parameter 
    """
    parameter_set = 0
    parameter_data = input_data.loc[(input_data.parameter == parameter)] # get the distribution of values for the specified parameter
    parameter_data = parameter_data.reset_index()
    np.random.seed(seed)
    random_parameter = np.random.randint(0,100)
    for i in range(len(parameter_data)):
        if i == 0:
            if random_parameter < parameter_data['value_for_input'][i]:
                parameter_set = parameter_data['value'][i]
                break
        else:
            if (random_parameter >= parameter_data['value_for_input'][i-1]) and (random_parameter <= parameter_data['value_for_input'][i]):
                parameter_set = parameter_data['value'][i]
                break
            else:
                continue
    return parameter_set


def get_flood_map_data(flood_map):
    """
    Getting the flood map characteristics.
    
    Parameters
    ----------
    flood_map: flood map in tif format

    Returns
    -------
    band, bound_l, bound_r, bound_t, bound_b: characteristics of the tif-file
    """
    band = flood_map.read(1)
    bound_l = flood_map.bounds.left
    bound_r = flood_map.bounds.right
    bound_t = flood_map.bounds.top
    bound_b = flood_map.bounds.bottom
    return band, bound_l, bound_r, bound_t, bound_b

shapefile_path = "input_data\\model_domain\\houston_model\\houston_model.shp"
floodplain_path = "input_data\\floodplain\\floodplain_area.shp"

#shapefile_path = "C:\\Users\\David\\Desktop\\final_project\\ABM-Project\\input_data\\model_domain\\houston_model\\houston_model.shp"
#floodplain_path = "C:\\Users\\David\\Desktop\\final_project\\ABM-Project\\input_data\\floodplain\\floodplain_area.shp"

# Model area setup
map_domain_gdf = gpd.GeoDataFrame.from_file(shapefile_path)
map_domain_gdf = map_domain_gdf.to_crs(epsg=26915)
map_domain_geoseries = map_domain_gdf['geometry']
map_minx, map_miny, map_maxx, map_maxy = map_domain_geoseries.total_bounds
map_domain_polygon = map_domain_geoseries[0]  # The geoseries contains only one polygon
prepare(map_domain_polygon)

# Floodplain setup
floodplain_gdf = gpd.GeoDataFrame.from_file(floodplain_path)
floodplain_gdf = floodplain_gdf.to_crs(epsg=26915)
floodplain_geoseries = floodplain_gdf['geometry']
floodplain_multipolygon = floodplain_geoseries[0]  # The geoseries contains only one multipolygon
prepare(floodplain_multipolygon)

def generate_random_location_within_map_domain():
    """
    Generate random location coordinates within the map domain polygon.

    Returns
    -------
    x, y: lists of location coordinates, longitude and latitude
    """
    while True:
        # generate random location coordinates within square area of map domain
        x = np.random.uniform(map_minx, map_maxx)
        y = np.random.uniform(map_miny, map_maxy)
        # check if the point is within the polygon, if so, return the coordinates
        if contains_xy(map_domain_polygon, x, y):
            return x, y

def get_flood_depth(corresponding_map, location, band):
    """ 
    To get the flood depth of a specific location within the model domain.
    Households are placed randomly on the map, so the distribution does not follow reality.
    
    Parameters
    ----------
    corresponding_map: flood map used
    location: household location (a Shapely Point) on the map
    band: band from the flood map

    Returns
    -------
    depth: flood depth at the given location
    """
    row, col = corresponding_map.index(location.x, location.y)
    depth = band[row -1, col -1]
    return depth
    

def get_position_flood(bound_l, bound_r, bound_t, bound_b, img, seed):
    """ 
    To generater the position on flood map for a household.
    Households are placed randomly on the map, so the distribution does not follow reality.
    
    Parameters
    ----------
    bound_l, bound_r, bound_t, bound_b, img: characteristics of the flood map data (.tif file)
    seed: seed to generate the location on the map

    Returns
    -------
    x, y: location on the map
    row, col: location within the tif-file
    """
    x = np.random.randint(round(bound_l, 0), round(bound_r, 0))
    y = np.random.randint(round(bound_b, 0), round(bound_t, 0))
    row, col = img.index(x, y)
    return x, y, row, col

def calculate_basic_flood_damage(agent, flood_depth):
    """
    To get flood damage based on flood depth of household
    from de Moer, Huizinga (2017) with logarithmic regression over it.
    If flood depth > 6m, damage = 1.
    
    Parameters
    ----------
    flood_depth : flood depth as given by location within model domain

    Returns
    -------
    flood_damage : damage factor between 0 and 1
    """
    if flood_depth >= 6:
        flood_depth = 6
    if (flood_depth-6*agent.total_adaptation_level) < 0.025:
        flood_damage = 0
    else:
        # see flood_damage.xlsx for function generation
        flood_damage = 0.1746 * math.log(flood_depth-6*agent.total_adaptation_level) + 0.6483
    return flood_damage * 235000

def calculate_influenced_risk_profile(model):
    influenced_risk_profile_vector = []
    for i in range(model.number_of_households):
        new_risk_profile = model.agents[i].risk_profile
        friends = model.agents[i].get_friends(1)
        for friend in friends:
            new_risk_profile += 0.5 / len(friends) * model.trust_matrix[i, friend] * (model.agents[friend].risk_profile - model.agents[i].risk_profile)
        
        if new_risk_profile > 1:
            new_risk_profile = 1
        if new_risk_profile < 0:
            new_risk_profile = 0

        influenced_risk_profile_vector.append(new_risk_profile)
    return influenced_risk_profile_vector

def generate_dictionary():
    results1 = {}
    results1["predicted total damage"] = [0, 0, 0, 0, 0, 0]
    results1["predicted average damage"] = [0, 0, 0, 0, 0, 0]
    results1["average adaptation level"] = [0, 0, 0, 0, 0, 0]
    results1["number of fully adapted agents"] = [0, 0, 0, 0, 0, 0]
    results1["average adaptation level per wealth"] = [0, 0, 0, 0, 0]
    results1["average damage per water depth"] = [0, 0, 0, 0, 0, 0]

    results2 = {}
    results2["predicted total damage"] = [0, 0, 0, 0, 0, 0]
    results2["predicted average damage"] = [0, 0, 0, 0, 0, 0]
    results2["average adaptation level"] = [0, 0, 0, 0, 0, 0]
    results2["number of fully adapted agents"] = [0, 0, 0, 0, 0, 0]
    results2["average adaptation level per wealth"] = [0, 0, 0, 0, 0]
    results2["average damage per water depth"] = [0, 0, 0, 0, 0, 0]

    results3 = {}
    results3["predicted total damage"] = [0, 0, 0, 0, 0, 0]
    results3["predicted average damage"] = [0, 0, 0, 0, 0, 0]
    results3["average adaptation level"] = [0, 0, 0, 0, 0, 0]
    results3["number of fully adapted agents"] = [0, 0, 0, 0, 0, 0]
    results3["average adaptation level per wealth"] = [0, 0, 0, 0, 0]
    results3["average damage per water depth"] = [0, 0, 0, 0, 0, 0]

    results4 = {}
    results4["predicted total damage"] = [0, 0, 0, 0, 0, 0]
    results4["predicted average damage"] = [0, 0, 0, 0, 0, 0]
    results4["average adaptation level"] = [0, 0, 0, 0, 0, 0]
    results4["number of fully adapted agents"] = [0, 0, 0, 0, 0, 0]
    results4["average adaptation level per wealth"] = [0, 0, 0, 0, 0]
    results4["average damage per water depth"] = [0, 0, 0, 0, 0, 0]

    results5 = {}
    results5["predicted total damage"] = [0, 0, 0, 0, 0, 0]
    results5["predicted average damage"] = [0, 0, 0, 0, 0, 0]
    results5["average adaptation level"] = [0, 0, 0, 0, 0, 0]
    results5["number of fully adapted agents"] = [0, 0, 0, 0, 0, 0]
    results5["average adaptation level per wealth"] = [0, 0, 0, 0, 0]
    results5["average damage per water depth"] = [0, 0, 0, 0, 0, 0]

    results6 = {}
    results6["predicted total damage"] = [0, 0, 0, 0, 0, 0]
    results6["predicted average damage"] = [0, 0, 0, 0, 0, 0]
    results6["average adaptation level"] = [0, 0, 0, 0, 0, 0]
    results6["number of fully adapted agents"] = [0, 0, 0, 0, 0, 0]
    results6["average adaptation level per wealth"] = [0, 0, 0, 0, 0]
    results6["average damage per water depth"] = [0, 0, 0, 0, 0, 0]

    return results1, results2, results3, results4, results5, results6


def save_data(dictionary, model, tick, number_runs):
    total_damage = 0
    for i in range(model.number_of_households):
        household = model.agents[i]
        total_damage += household.flood_damage_estimated
    dictionary["predicted total damage"][tick] += total_damage/number_runs
    dictionary["predicted average damage"][tick] += total_damage/model.number_of_households/number_runs

    average_adaptation_level = 0
    for i in range(model.number_of_households):
        household = model.agents[i]
        average_adaptation_level += household.total_adaptation_level
    dictionary["average adaptation level"][tick] += average_adaptation_level/model.number_of_households/number_runs
    
    number_of_adapted = 0
    for i in range(model.number_of_households):
        household = model.agents[i]
        if household.total_adaptation_level >= household.flood_depth_estimated:
            number_of_adapted += 1
    dictionary["number of fully adapted agents"][tick] += number_of_adapted/number_runs

    return dictionary


def save_result_data(dictionary, model, tick, number_runs):
    for i in range(5):
        adaptation_level = 0
        number_agents = 0
        for j in range(model.number_of_households):
            if model.agents[j].wealth_type == i:
                adaptation_level += model.agents[j].total_adaptation_level
                number_agents =+ 1
        if number_agents > 0:
            dictionary["average adaptation level per wealth"][i] += adaptation_level/number_agents/number_runs
        else:
            dictionary["average adaptation level per wealth"][i] += 0
    
    for i in range(6):
        damage_level = 0
        number_agents = 0
        for j in range(model.number_of_households):
            if model.agents[j].flood_depth_actual >= i and model.agents[j].flood_depth_actual < (i+1):
                damage_level += model.agents[j].flood_damage_actual
                number_agents += 1
        if number_agents > 0:
            dictionary["average damage per water depth"][i] += damage_level/number_agents/number_runs
        else:
            dictionary["average damage per water depth"][i] = 0
    return dictionary


def create_graphs(dictionary, folder_name, plot_names, x_names, y_names):
    for i in range(6):
        create_plot(dictionary[plot_names[i]], folder_name, plot_names[i], x_names[i], y_names[i])


def create_plot(vector, save_folder, file_name, x_axis_name, y_axis_name):
    plt.clf()
    # Create x values for the plot
    x_values = list(range(1, len(vector) + 1))
    
    # Plot the vector
    plt.plot(x_values, vector, marker='o')
    
    # Set x-axis ticks
    plt.xticks(x_values)
    
    # Set labels
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(file_name)
    
    # Ensure folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save the plot as PNG
    save_path = os.path.join(save_folder, file_name + '.png')
    plt.savefig(save_path)
