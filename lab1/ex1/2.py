def neuralNetwork(inp, weight):
     prediction = inp * weight
     return prediction
out_1 = neuralNetwork(150, 0.3)
out_2 = neuralNetwork(130, 0.4)
print(out_1)
print(out_2)