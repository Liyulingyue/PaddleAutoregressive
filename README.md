# PaddleAutoregressive

基于飞桨实现自回归模型
Implement the Autoregressive with PaddlePaddle

## About Autoregressive
## 关于自回归模型

参考[知乎页面](https://zhuanlan.zhihu.com/p/406360992)，几个自回归模型的可以看做满足以下格式：

$A(p)y(k) = B(q)u(k) + C(o)v(k)$

其中：
- $A(p)y(k) = y(k) + a_1y(k-1) + a_2y(k-2) + ... + a_py(k-p)$，$B(q)u(k)$和$C(o)v(k)$类似。
- $y$是因变量
- $u$是自变量
- $v$是扰动项。

特别的，对于最基础的AR模型，可设$B$和$C$为0，即

$A(p)y(k) = 0$

$\Rightarrow A(p)y(k) = y(k) + a_1y(k-1) + a_2y(k-2) + ... + a_py(k-p)$

$\Rightarrow y(k) = -a_1y(k-1) - a_2y(k-2) - ... - a_py(k-p)$

因而，我们可以实现一个通用的基础模块，并在这个基础模块的基础上不断封装从而实现AR、ARMA、FIR等时间预测模型。

## How to use
## 如何使用

```commandline
# 下载代码
! git clone https://github.com/Liyulingyue/PaddleAutoregressive.git
%cd ~/PaddleAutoregressive
# 安装
! pip install -e .
```

安装后即可自由地和Paddle中实现过的网络进行组网。

```commandline
%cd ~
import PaddleAutoregressive.AR as AR
import paddle
model = AR.AR(5)
paddle.summary(model,(5,5))
```
