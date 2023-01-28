from PaddleAutoregressive.Autoregressive import Autoregressive
import paddle

class AR(paddle.nn.Layer):
    def __init__(self, y_features):
        # y_features should be a int nunber
        # x_features should be a list of int
        # e_features should be a int nunber
        super(AR, self).__init__()
        self.y_features = y_features
        self.Autoregressive = Autoregressive(y_features, [], 0)

    def forward(self, *inputs):
        output = self.Autoregressive(inputs[0])
        return output
