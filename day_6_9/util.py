
from typing import Literal, Any

#TODO 统一的打印函数
# 定义颜色类型
ColorType = Literal['green', 'yellow', 'red', 'reset']

def pt(
    label: str = "结果",      # 默认标签
    ot: Any = None,          # 默认空结果
    color: ColorType = 'green'  # 默认绿色
) -> None:
    """统一的打印函数
    :param color: 颜色选项，可以是 'green', 'yellow', 'red' 或 'reset'
    :param label: 显示的标签文本
    :param ot: 要打印的结果
    """
    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'reset': '\033[0m'
    }
    print(f"{colors[color]}{label}:{colors['reset']}\n {ot}")
