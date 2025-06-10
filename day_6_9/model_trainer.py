import numpy as np
from main_Lib_1 import neural_network
from model_saver import save_model, load_model, create_model_params, get_model_info

class PowerActivationNetworkTrainer:
    """
    幂次激活神经网络训练器
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dims, 
                 output_dim, 
                 activation_degrees=None,
                 activation_function=None,
                 temperature=1.0):
        """
        初始化训练器
        
        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            activation_degrees: 每层激活函数的最高次幂
            activation_function: 激活函数
            temperature: softmax温度参数
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.temperature = temperature
        
        # 默认使用power函数作为激活函数
        self.activation_function = activation_function if activation_function else lambda x, d: x**d
        
        # 创建网络结构
        all_dims = [input_dim] + hidden_dims + [output_dim]
        
        # 如果未指定激活度数，则默认为2（即使用线性和二次项）
        if activation_degrees is None:
            self.activation_degrees = [2] * len(all_dims)
        else:
            self.activation_degrees = activation_degrees
        
        # 初始化权重、激活权重和偏置
        self.weights = []
        self.activation_weights = []
        self.biases = []
        
        for i in range(len(all_dims) - 1):
            # 权重矩阵 (输出维度 x 输入维度)
            W = np.random.randn(all_dims[i+1], all_dims[i]) * 0.1
            self.weights.append(W)
            
            # 激活权重 (输出维度 x (激活度数+1))
            # +1是因为包括0次幂(常数项)
            AW = np.random.randn(all_dims[i+1], self.activation_degrees[i] + 1) * 0.1
            self.activation_weights.append(AW)
            
            # 偏置向量 (输出维度 x 1)
            b = np.zeros((all_dims[i+1], 1))
            self.biases.append(b)
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入数据 (样本数 x 输入维度)
        
        返回:
            输出概率 (样本数 x 输出维度)
        """
        # 确保X是二维的
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            raise ValueError("输入数据必须是1维或2维")
        
        # 如果X是(样本数, 特征数)，转置为(特征数, 样本数)
        if X.shape[1] == self.input_dim:
            X = X.T
        
        outputs = []
        # 对每个样本进行前向传播
        for i in range(X.shape[1]):
            x = X[:, i:i+1]  # 获取第i个样本作为列向量
            y = neural_network(
                x, 
                self.weights, 
                self.activation_weights, 
                self.biases, 
                self.activation_function,
                self.temperature
            )
            outputs.append(y.T)  # 转置使得输出为(1, 输出维度)
        
        # 将所有输出堆叠为(样本数, 输出维度)的矩阵
        return np.vstack(outputs)
    
    def predict(self, X):
        """
        预测类别
        
        参数:
            X: 输入数据
        
        返回:
            预测的类别索引
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def save(self, filepath, format='pickle', extra_params=None):
        """
        保存模型到文件
        
        参数:
            filepath: 文件保存路径
            format: 保存格式
            extra_params: 额外参数
        """
        model_params = create_model_params(
            self.weights,
            self.activation_weights,
            self.biases,
            self.temperature,
            extra_params
        )
        
        # 添加网络结构信息
        model_params['input_dim'] = self.input_dim
        model_params['hidden_dims'] = self.hidden_dims
        model_params['output_dim'] = self.output_dim
        model_params['activation_degrees'] = self.activation_degrees
        
        save_model(model_params, filepath, format)
    
    @classmethod
    def load(cls, filepath, format='pickle'):
        """
        从文件加载模型
        
        参数:
            filepath: 文件路径
            format: 文件格式
        
        返回:
            加载的模型实例
        """
        model_params = load_model(filepath, format)
        
        # 创建模型实例
        model = cls(
            input_dim=model_params['input_dim'],
            hidden_dims=model_params['hidden_dims'],
            output_dim=model_params['output_dim'],
            activation_degrees=model_params['activation_degrees'],
            temperature=model_params['temperature']
        )
        
        # 替换权重和偏置
        model.weights = model_params['weights']
        model.activation_weights = model_params['activation_weights']
        model.biases = model_params['biases']
        
        return model
    
    def get_model_info(self):
        """
        获取模型信息
        
        返回:
            模型信息字符串
        """
        model_params = {
            'weights': self.weights,
            'activation_weights': self.activation_weights,
            'biases': self.biases
        }
        
        info = [
            f"输入维度: {self.input_dim}",
            f"隐藏层维度: {self.hidden_dims}",
            f"输出维度: {self.output_dim}",
            f"激活度数: {self.activation_degrees}",
            f"温度参数: {self.temperature}",
            get_model_info(model_params)
        ]
        
        return "\n".join(info) 