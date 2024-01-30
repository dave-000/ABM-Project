from model.model import AdaptationModel

model = AdaptationModel()

for i in range(5):
    print(i)
    model.step()
