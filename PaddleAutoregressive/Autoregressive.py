import paddle

class Autoregressive(paddle.nn.Layer):
    def __init__(self, y_features, x_features, e_features):
        # y_features should be a int nunber
        # x_features should be a list of int
        # e_features should be a int nunber
        super(Autoregressive, self).__init__()

        # if not isinstance(in_features, int): raise ValueError('Wrong in_features')

        self.y_features = y_features
        self.x_features = x_features
        self.e_features = e_features

        linear_list = []
        if y_features != 0:
            linear_list.append(paddle.nn.Linear(y_features, 1, bias_attr=True))
        for _x in x_features:
            linear_list.append(paddle.nn.Linear(_x, 1, bias_attr=True))
        if e_features != 0:
            linear_list.append(paddle.nn.Linear(e_features, 1, bias_attr=True))
        self.linear_list = paddle.nn.Sequential(*linear_list)

    def forward(self, *inputs):
        output = paddle.to_tensor([0]).astype('float32')
        for i in range(len(self.linear_list)):
            output += self.linear_list[i](inputs[i])
        return output