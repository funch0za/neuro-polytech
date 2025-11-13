def network_hidden(inp, weight_i):
    predict = [0] * len(weight_i)
    for i in range(len(weight_i)):
        ws = 0
        for j in range(len(inp)):
            ws += inp[j] * weight_i[i][j]
        predict[i] = ws
    return 

def network(inp, weight):
    predict_hidden = [0] * len(weight[0])
    for i in range(len(weight[0])):
        ws = 0
        for j in range(len(inp)):
            ws += inp[j] * weight[0][i][j]
        predict_hidden[i] = ws
    predict = [0] * len(weight[1])
    for i in range(len(weight[1])):
        ws = 0
        for j in range(len(predict_hidden)):
            ws += predict_hidden[j] * weight[0][i][j]
        predict[i] = ws
    return predict

inp = [23, 45]
weight_h_1 = [0.4, 0.1] 
weight_h_2 = [0.3, 0.2]
weight_out_1 = [0.4, 0.1] 
weight_out_2 = [0.3, 0.1] 
weights_h = [weight_h_1, weight_h_2]
weights_out = [weight_out_1, weight_out_2]
weights = [weights_h, weights_out]
print(network(inp, weights))