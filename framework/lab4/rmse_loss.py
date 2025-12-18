from layer import Layer

class RMSELoss(Layer):
    def forward(self, prediction, true_prediction):
        square_diff = (true_prediction - prediction) * (true_prediction - prediction)
        square_diff.data = np.full(square_diff.data.shape, np.sqrt(np.mean(square_diff.data)))
        return square_diff.__sum__(0)
