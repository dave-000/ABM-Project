from model.model import AdaptationModel

random_seed = 420
households_number = 25
flood_map = 'harvey'
network_type = 'watts_strogatz'
prob_network_connection = 0.4
num_of_edges = 3
num_of_nearest_neighbours = 5
government_type1 = "autocratic"
government_type2 = "democratic"

#Creates the two models
model_autocratic = AdaptationModel(seed=random_seed, number_of_households=households_number, flood_map_choice=flood_map, network=network_type, probability_of_network_connection=prob_network_connection, number_of_edges=num_of_edges, number_of_nearest_neighbours=num_of_nearest_neighbours, government_type=government_type1)
model_democratic = AdaptationModel(seed=random_seed, number_of_households=households_number, flood_map_choice=flood_map, network=network_type, probability_of_network_connection=prob_network_connection, number_of_edges=num_of_edges, number_of_nearest_neighbours=num_of_nearest_neighbours, government_type=government_type2)

for i in range(5):
    print("iteration number " + str(i))
    model_democratic.step()
