import numpy as np
import pickle

class PowerActivationNetwork:
    def __init__(self, input_size=784, hidden_size=50, output_size=10, max_power=3):
        """
        初始化幂激活神经网络
        
        参数:
        input_size: 输入维度 (MNIST为28x28=784)
        hidden_size: 隐藏层神经元数量
        output_size: 输出维度 (数字0-9共10类)
        max_power: 激活函数的最大幂次
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_power = max_power
        
        # 初始化权重和偏置，使用更小的scale以提高数值稳定性
        scale = 0.001
        self.weights = scale * np.random.randn(hidden_size, input_size)
        self.biases = np.zeros((hidden_size, 1))
        self.activation_weights = scale * np.random.randn(output_size, max_power, hidden_size)
        
        # 学习率
        self.learning_rate = 0.01
        
        # 预计算幂次指数
        self.powers = np.arange(1, max_power + 1)
    
    def vector_power(self, vector, max_power=None):
        """
        计算向量的1到max_power次幂
        
        参数:
        vector: 输入向量
        max_power: 最大幂次，默认使用网络配置的max_power
        
        返回:
        范德蒙德矩阵 [max_power, vector_length]
        """
        if max_power is None:
            max_power = self.max_power
            powers = self.powers
        else:
            powers = np.arange(1, max_power + 1)
        
        # 限制输入值范围，防止幂运算爆炸
        vector = np.clip(vector, -3.0, 3.0)
        
        # 高效的向量化计算
        if vector.ndim == 2:  # 如果是矩阵 [hidden_size, batch_size]
            # 转置以便广播 [batch_size, hidden_size]
            transposed = vector.T
            # 计算所有幂次 [batch_size, hidden_size, max_power]
            powered = np.power(transposed[:, :, np.newaxis], powers)
            # 转置回原始形状 [max_power, hidden_size, batch_size]
            return np.moveaxis(powered, 2, 0)
        else:  # 单个样本 [hidden_size, 1]
            # [max_power, hidden_size]
            return np.power(vector.T, powers[:, np.newaxis])
    
    def propagate(self, vector, weights, activation_weights, biases):
        """
        前向传播过程 (向量化实现)
        
        参数:
        vector: 输入向量 [input_size, batch_size]
        weights: 权重矩阵 [hidden_size, input_size]
        activation_weights: 激活权重 [output_size, max_power, hidden_size]
        biases: 偏置向量 [hidden_size, 1]
        
        返回:
        output_vector: 输出向量 [output_size, batch_size]
        linear_output: 线性输出 [hidden_size, batch_size]
        vandermonde_matrix: 范德蒙德矩阵 [max_power, hidden_size, batch_size]
        """
        # 线性变换 [hidden_size, batch_size]
        linear_output = weights @ vector
        if vector.ndim == 2:
            # 广播biases到所有样本
            linear_output += biases
        else:
            linear_output += biases
        
        # 计算幂次激活 [max_power, hidden_size, batch_size]
        vandermonde_matrix = self.vector_power(linear_output)
        
        # 优化的输出层计算 (批量处理)
        if vector.ndim == 2:  # 批处理情况
            batch_size = vector.shape[1]
            # 初始化输出 [output_size, batch_size]
            output_vector = np.zeros((self.output_size, batch_size))
            
            # 对每个样本计算
            for b in range(batch_size):
                # [max_power, hidden_size]
                vm_sample = vandermonde_matrix[:, :, b]
                
                # 高效矩阵乘法替代三重循环
                # [output_size, max_power, hidden_size] @ [max_power, hidden_size] -> [output_size]
                for i in range(self.output_size):
                    # 展平权重和vm以便点积
                    flat_weights = activation_weights[i].reshape(-1)
                    flat_vm = vm_sample.reshape(-1)
                    output_vector[i, b] = np.dot(flat_weights, flat_vm)
        else:  # 单样本情况
            # 初始化输出 [output_size, 1]
            output_vector = np.zeros((self.output_size, 1))
            
            # 高效矩阵乘法替代三重循环
            for i in range(self.output_size):
                # 展平权重和vm以便点积
                flat_weights = activation_weights[i].reshape(-1)
                flat_vm = vandermonde_matrix.reshape(-1)
                output_vector[i, 0] = np.dot(flat_weights, flat_vm)
        
        return output_vector, linear_output, vandermonde_matrix
    
    def softmax(self, x, temperature=1.0):
        """
        Softmax函数实现 (具有数值稳定性)
        
        参数:
        x: 输入向量或矩阵 [output_size, batch_size]
        temperature: 温度参数，控制分布的平滑度
        
        返回:
        softmax概率 [output_size, batch_size]
        """
        # 处理无效值
        x = np.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # 数值稳定性技巧
        x_safe = np.clip(x, -1e2, 1e2)  # 防止exp爆炸
        
        if x.ndim == 2:  # 批处理情况
            # 对每个样本找最大值
            max_x = np.max(x_safe, axis=0, keepdims=True)
            exp_x = np.exp((x_safe - max_x) / temperature)
            sum_exp = np.sum(exp_x, axis=0, keepdims=True)
            # 防止除零
            sum_exp = np.maximum(sum_exp, 1e-10)
            return exp_x / sum_exp
        else:
            max_x = np.max(x_safe)
            exp_x = np.exp((x_safe - max_x) / temperature)
            sum_exp = np.sum(exp_x)
            # 防止除零
            if sum_exp < 1e-10:
                return np.ones_like(x) / len(x)
            return exp_x / sum_exp
    
    def predict(self, x):
        """
        预测函数
        
        参数:
        x: 输入数据，可以是单个样本或批量样本
        
        返回:
        probabilities: 预测概率
        predicted_class: 预测类别
        """
        # 确保输入是正确的形状
        if x.ndim == 4:  # 如果是(batch_size, 1, height, width)
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1).T  # [input_size, batch_size]
        elif x.ndim == 3:  # 如果是(1, height, width)
            x = x.reshape(-1, 1)  # [input_size, 1]
        elif x.ndim == 2 and x.shape[1] > 1:  # 如果是(batch_size, input_size)
            x = x.T  # [input_size, batch_size]
        
        # 前向传播
        output_vector, _, _ = self.propagate(x, self.weights, self.activation_weights, self.biases)
        
        # 应用softmax
        probabilities = self.softmax(output_vector)
        
        # 获取预测类别
        if probabilities.ndim == 2:  # 批处理情况
            predicted_class = np.argmax(probabilities, axis=0)
        else:
            predicted_class = np.argmax(probabilities)
        
        return probabilities, predicted_class
    
    def update_weights_batch(self, x_batch, t_batch):
        """
        批量更新网络权重
        
        参数:
        x_batch: 输入数据批次 [batch_size, channels, height, width] 或 [batch_size, input_size]
        t_batch: 目标标签批次 [batch_size]
        """
        batch_size = len(t_batch)
        
        # 确保输入是正确的形状
        if x_batch.ndim == 4:  # 如果是(batch_size, 1, height, width)
            x_batch = x_batch.reshape(batch_size, -1).T  # [input_size, batch_size]
        elif x_batch.ndim == 2 and x_batch.shape[0] == batch_size:
            x_batch = x_batch.T  # [input_size, batch_size]
        
        # 前向传播
        output_vector, linear_output, vandermonde_matrix = self.propagate(
            x_batch, self.weights, self.activation_weights, self.biases
        )
        
        # 计算当前输出概率 [output_size, batch_size]
        probabilities = self.softmax(output_vector)
        
        # 创建目标概率分布 (one-hot编码) [output_size, batch_size]
        target = np.zeros_like(probabilities)
        for i, t in enumerate(t_batch):
            target[t, i] = 1.0
        
        # 计算误差 [output_size, batch_size]
        error = target - probabilities
        
        # 初始化梯度
        dactivation_weights = np.zeros_like(self.activation_weights)
        dweights = np.zeros_like(self.weights)
        dbiases = np.zeros_like(self.biases)
        
        # 更新激活权重 (对每个样本计算梯度并累积)
        for b in range(batch_size):
            # 当前样本的误差 [output_size]
            error_sample = error[:, b:b+1]
            
            # 当前样本的范德蒙德矩阵 [max_power, hidden_size]
            vm_sample = vandermonde_matrix[:, :, b]
            
            # 更新激活权重 - 梯度: error * vandermonde_matrix
            for i in range(self.output_size):
                for j in range(self.max_power):
                    for k in range(self.hidden_size):
                        dactivation_weights[i, j, k] += error_sample[i, 0] * vm_sample[j, k]
            
            # 计算隐藏层误差 (关于线性输出的梯度)
            hidden_error = np.zeros((self.hidden_size, 1))
            for i in range(self.output_size):
                for j in range(self.max_power):
                    for k in range(self.hidden_size):
                        # 计算梯度: error * 激活权重 * 导数(power函数)
                        if linear_output[k, b] != 0:  # 避免0的幂次导数
                            derivative = (j + 1) * (linear_output[k, b] ** j)
                            hidden_error[k] += error_sample[i, 0] * self.activation_weights[i, j, k] * derivative
            
            # 更新权重和偏置 - 梯度: hidden_error * x
            x_sample = x_batch[:, b:b+1]
            dweights += hidden_error @ x_sample.T
            dbiases += hidden_error
        
        # 应用梯度 (取平均)
        self.activation_weights += self.learning_rate * (dactivation_weights / batch_size)
        self.weights += self.learning_rate * (dweights / batch_size)
        self.biases += self.learning_rate * (dbiases / batch_size)
    
    def update_weights(self, x, correct_label):
        """
        根据单个样本更新权重 (兼容旧接口)
        
        参数:
        x: 输入数据
        correct_label: 正确的标签 (0-9)
        """
        # 确保x是正确的形状
        if x.ndim == 4:  # 如果是(batch_size, 1, height, width)
            x = x.reshape(-1, 1)
        elif x.ndim == 3:  # 如果是(1, height, width)
            x = x.reshape(-1, 1)
        elif x.ndim == 2 and x.shape[1] == 1:
            pass  # 已经是正确形状 [input_size, 1]
        else:
            raise ValueError(f"不支持的输入形状: {x.shape}")
        
        # 使用批量更新函数处理单个样本
        self.update_weights_batch(x.T, np.array([correct_label]))
    
    def save_params(self, file_name="power_network_params.pkl"):
        """保存网络参数"""
        params = {
            'weights': self.weights,
            'biases': self.biases,
            'activation_weights': self.activation_weights,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'max_power': self.max_power,
            'learning_rate': self.learning_rate
        }
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    
    def load_params(self, file_name="power_network_params.pkl"):
        """加载网络参数"""
        try:
            with open(file_name, 'rb') as f:
                params = pickle.load(f)
            
            self.weights = params['weights']
            self.biases = params['biases']
            self.activation_weights = params['activation_weights']
            self.input_size = params['input_size']
            self.hidden_size = params['hidden_size']
            self.output_size = params['output_size']
            self.max_power = params['max_power']
            self.learning_rate = params['learning_rate']
            # 重新计算幂次指数
            self.powers = np.arange(1, self.max_power + 1)
            return True
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            return False