import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

input_ids = torch.tensor([2,3,5,1])
vocal_size = 6                      #代表词表大小
output_dim = 3                      # 嵌入维度（每个词用3个数字表示）
torch.manual_seed(123)              #固定随机数种子。因为 PyTorch 在创建嵌入层时会默认随机初始化里面的数字。固定种子为 123，能确保你每次运行、或者在别人电脑上运行，得到的随机数矩阵都完全一模一样，方便调试。
embedding_lay = torch.nn.Embedding(vocal_size,output_dim)
#：正式实例化一个嵌入层。此时，PyTorch 会在内存中悄悄建立一个形状为 [6, 3] 的大查找表（Weight Matrix）。这个表有 6 行（对应 6 个词），3 列（对应每个词的 3 维特征）。
print(embedding_lay.weight)
# 打印权重矩阵
print(embedding_lay(input_ids))