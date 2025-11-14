def network(inp, weight):
    predict = [0] * len(weight)
    for i in range(len(predict)):
        predict[i] = inp * weight[i]
    return predict

out1 = network(4, [0.2, 0.5])
print(out1)