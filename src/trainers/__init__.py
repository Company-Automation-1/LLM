# 训练器模块初始化文件
from src.trainers.backprop import (diff_vect_weight, diff_vect_aweight, 
                                 diff_vect_bias, gradient)

__all__ = ['diff_vect_weight', 'diff_vect_aweight', 'diff_vect_bias', 'gradient'] 