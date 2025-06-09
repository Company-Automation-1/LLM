import numpy as np
import main_Lib_1 as Lib1
import main_Lib_2 as Lib2
import trainer

def test():
    """
    测试函数
    """
    input_vector = np.array([
        [1, 2, 3]
        ]).T
    weights_1 = np.array([
        [2, 2, 1], 
        [2, 2, 1], 
        [1, 1, 1]
        ])
    activation_weights_1 = np.array([
        [4, 3, 5], 
        [2, 1, 2], 
        [1, 2, 6], 
        [3, 2, 4]
        ])
    biases_1 = np.array([
        [2, 2, 1]
        ]).T
    
    # 组织为层参数列表
    weights = [weights_1]
    activation_weights = [activation_weights_1]
    biases = [biases_1]

    output = Lib1.neural_network(input_vector, weights, activation_weights, biases, Lib1.power, 1)
    # print(output)

#TODO: 执行测试
if __name__ == "__main__":
    # test()
    print(Lib1.softplus(1))