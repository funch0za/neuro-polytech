def network(inp, weight):
    predict = 0
    for i in range(len(weight)):
        predict += weight[i] * inp[i]
    return predict
out1 = network([150, 40], [0.3, 0.4])
out2 = network([80, 60], [0.2, 0.4])
print(out1, out2)

