def neuralNetwork(inp, weight):
     prediction = inp * weight
     return prediction
out_1 = neuralNetwork(110, 0.9)
out_2 = neuralNetwork(10, 0.2)
print(out_1)
print(out_2)