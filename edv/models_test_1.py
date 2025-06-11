import numpy as np
import os
import sys
from main_Lib_1 import power, vandermonde_matrix, diagonal_product, propagate, softmax, neural_network, diff_softmax
from main_Lib_2 import softplus, sigmoid
from util import pt

from traner_test import diff_vect_aweight, diff_vect_weight, diff_vect_bias, gradient
from model_saver import load_model, save_model

# model_data = {
#   'a' : a,
#   'b' : b,
#   'vectors': vectors,
#   'activation_weights': activation_weights,
# }

# # 确保models目录存在
# os.makedirs("models", exist_ok=True)

# # 保存模型示例
# save_model(model_data, "models/demo_1.pkl")

# 加载模型示例
# loaded_model = load_model("model/999.pkl")

# 打印模型信息
# pt('模型包含的键:', list(loaded_model.keys()))
# pt('模型信息:', list(loaded_model.values()))
# pt('vectors', loaded_model['vectors'])
# pt('activation_vectors', loaded_model['intermediate_vectors'])

# vector = np.array([[1, 2, 3, 4]]).T
# diff_vector = np.array([[1, 1, 1, 2]]).T


# pt('diff_softmax', diff_softmax(vector, diff_vector, 1))