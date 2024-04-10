from model.model import AdaptationModel
from model.functions import generate_dictionary, save_data, save_result_data, create_graphs

random_seed = 42
households_number = 25
flood_map = 'harvey'
network_type = 'watts_strogatz'
prob_network_connection = 0.4
num_of_edges = 3
num_of_nearest_neighbours = 5

#NOTE add possibility to have more control over the policies

results1, results2, results3, results4, results5, results6 = generate_dictionary()

number_of_runs = 3

for i in range(number_of_runs):
    #Creates the six scenarios
    model_01_no_policies    = AdaptationModel(seed=random_seed, number_of_households=households_number, flood_map_choice=flood_map, network=network_type, probability_of_network_connection=prob_network_connection, number_of_edges=num_of_edges, number_of_nearest_neighbours=num_of_nearest_neighbours, subsidies_type="none", psa_type="none")
    model_02_subs_weak      = AdaptationModel(seed=random_seed, number_of_households=households_number, flood_map_choice=flood_map, network=network_type, probability_of_network_connection=prob_network_connection, number_of_edges=num_of_edges, number_of_nearest_neighbours=num_of_nearest_neighbours, subsidies_type="weak", psa_type="none")
    model_03_subs_strong    = AdaptationModel(seed=random_seed, number_of_households=households_number, flood_map_choice=flood_map, network=network_type, probability_of_network_connection=prob_network_connection, number_of_edges=num_of_edges, number_of_nearest_neighbours=num_of_nearest_neighbours, subsidies_type="strong", psa_type="none")
    model_04_psa_weak       = AdaptationModel(seed=random_seed, number_of_households=households_number, flood_map_choice=flood_map, network=network_type, probability_of_network_connection=prob_network_connection, number_of_edges=num_of_edges, number_of_nearest_neighbours=num_of_nearest_neighbours, subsidies_type="none", psa_type="weak")
    model_05_psa_strong     = AdaptationModel(seed=random_seed, number_of_households=households_number, flood_map_choice=flood_map, network=network_type, probability_of_network_connection=prob_network_connection, number_of_edges=num_of_edges, number_of_nearest_neighbours=num_of_nearest_neighbours, subsidies_type="none", psa_type="strong")
    model_06_subs_psa_weak  = AdaptationModel(seed=random_seed, number_of_households=households_number, flood_map_choice=flood_map, network=network_type, probability_of_network_connection=prob_network_connection, number_of_edges=num_of_edges, number_of_nearest_neighbours=num_of_nearest_neighbours, subsidies_type="weak", psa_type="weak")


    for i in range(6):
        print("iteration number " + str(i))
        model_01_no_policies.step()
        model_02_subs_weak.step()
        model_03_subs_strong.step()
        model_04_psa_weak.step()
        model_05_psa_strong.step()
        model_06_subs_psa_weak.step()
        results1 = save_data(results1, model_01_no_policies, tick=i, number_runs=number_of_runs)
        results2 = save_data(results2, model_02_subs_weak, tick=i, number_runs=number_of_runs)
        results3 = save_data(results3, model_03_subs_strong, tick=i, number_runs=number_of_runs)
        results4 = save_data(results4, model_04_psa_weak, tick=i, number_runs=number_of_runs)
        results5 = save_data(results5, model_05_psa_strong, tick=i, number_runs=number_of_runs)
        results6 = save_data(results6, model_06_subs_psa_weak, tick=i, number_runs=number_of_runs)

results1 = save_result_data(results1, model_01_no_policies, tick=i, number_runs=number_of_runs)
results2 = save_result_data(results2, model_02_subs_weak, tick=i, number_runs=number_of_runs)
results3 = save_result_data(results3, model_03_subs_strong, tick=i, number_runs=number_of_runs)
results4 = save_result_data(results4, model_04_psa_weak, tick=i, number_runs=number_of_runs)
results5 = save_result_data(results5, model_05_psa_strong, tick=i, number_runs=number_of_runs)
results6 = save_result_data(results6, model_06_subs_psa_weak, tick=i, number_runs=number_of_runs)


plot_names = ["predicted total damage", "predicted average damage", "average adaptation level", "number of fully adapted agents", "average adaptation level per wealth", "average damage per water depth"]
x_names = ["tick", "tick", "tick", "tick", "wealth value", "water depth actual"]
y_names = ["predicted total damage [$]", "predicted average damage [$]", "average adaptation level", "number of fully adapted agents", "average adaptation level", "actual damage [$]"]
create_graphs(results1, "model\\plots\\no_policy", plot_names, x_names, y_names)
create_graphs(results2, "model\\plots\\weak_subsidies_policy", plot_names, x_names, y_names)
create_graphs(results3, "model\\plots\\strong_subsidies_policy", plot_names, x_names, y_names)
create_graphs(results4, "model\\plots\\weak_psa_policy", plot_names, x_names, y_names)
create_graphs(results5, "model\\plots\\strong_psa_policy", plot_names, x_names, y_names)
create_graphs(results6, "model\\plots\\weak_psa_subsidies_policy", plot_names, x_names, y_names)
