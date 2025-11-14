def network(inp, weight):
    predict = [0] * len(weight)
    for i in range(len(weight)):
        predict[i] = sum([inp[j] * weight[i][j] for j in range(len(inp))])
    return predict

print("1")
print(network([50, 165], [[0.2, 0.1], [0.3, 0.1]]))
print("2")
print(network([50, 165, 45], [[0.2, 0.1, 0.65], [0.3, 0.1, 0.7]]))
print("3")
print(network([50, 165, 45], [[0.2, 0.1, 0.65], [0.3, 0.1, 0.7], [0.5, 0.4, 0.34]]))
