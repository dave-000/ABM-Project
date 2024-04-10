# Builds the trust matrix
self.trust_matrix = np.random.normal(0.5, 0.4, size = (number_of_households, number_of_households))
for i in range(number_of_households):
    for j in range(number_of_households):
        if self.trust_matrix[i,j] > 1:
            self.trust_matrix[i,j] = 1
        if self.trust_matrix[i,j] < -1:
            self.trust_matrix[i,j] = -1

#Calculates the updated risk profile (NOTE, the risk profile are updated with the new values only at the END of the step)
self.next_risk_profile = calculate_influenced_risk_profile(self)

#Updates with the new values
for i in range(self.number_of_households):
    self.agents[i].risk_profile = self.next_risk_profile[i]



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