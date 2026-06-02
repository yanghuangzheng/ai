import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import tiktoken


from torch.utils.data import Dataset,DataLoader
#下面的(Dataset)起到了继承的作用 如果你没写 (Dataset)：即使你里面写了 __len__ 和 __getitem__，PyTorch 依然会认为你这只是个普通的 Python 类，直接拒绝 cooperative 工作并报错。
class GPTDatasetV1(Dataset):        
    def __init__(self,txt,tokenizer,max_length,stride):
          self.input_ids=[]
          self.target_ids=[]
           # 将原始文本通过分词器编码为 Token ID 列表
          token_ids=tokenizer.encode(txt)
            # 使用滑动窗口切分文本                    
          for i in range(0,len(token_ids) - max_length,stride): #stride 步长
               input_chunk = token_ids[i:i+max_length]
               target_chunk=token_ids[i+1:i+max_length+1]
                # 将切分好的片段转为 PyTorch 张量并保存
               self.input_ids.append((torch.tensor(input_chunk)))
               self.target_ids.append((torch.tensor(target_chunk)))
    def __len__(self):
         return len(self.input_ids)
    
    def __getitem__(self, idx):
         return self.input_ids[idx],self.target_ids[idx]
#####################################################################################
def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
         tokenizer = tiktoken.get_encoding("gpt2")
         dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)
         #dataset 对象在内存中必须具备以下三个硬性特征
         #它身上带有两个列表（通常在 __init__ 中生成），分别存储了全部切分好的张量：
         #self.input_ids：一堆形如 tensor([]) 的 1 维张量列表。self.target_ids：一堆形如 tensor([]) 的 1 维张量标签列表。
         #特征二：知道自己的总数它必须实现 __len__ 方法，能向外汇报它到底切出了多少个这样的片段。
         #特征三：支持通过索引取货它必须实现 __getitem__ 方法，能通过 dataset 这种下标语法，精准返回第idx个片段的 (input_tensor, target_tensor)。
         dataloader = DataLoader(       #当 DataLoader 启动并开始干活时，它在底层主要调用了 dataset 内部的两个核心魔法方法：__len__ 和 __getitem__
              dataset,                  #返回一个 Python 元组（Tuple），元组内部包裹着 PyTorch 张量（Tensor）。
              batch_size=batch_size,    #第一步：调用 dataset.__len__()
              shuffle=shuffle,          #第二步：疯狂调用 dataset.__getitem__(idx)
              drop_last=drop_last,
              num_workers=num_workers,
         )                                  
         return dataloader
#把做好的面包原料（`dataset`）放进自动化传送带（`DataLoader`），配置好传送规则。
### 2. 关键参数的通俗含义
#DataLoader` 括号里配置的这几个参数，决定了数据怎么喂给 GPT 模型：
#| 参数名称 | 默认值 | 通俗解释与作用 |
#| :--- | :--- | :--- |
#| **`batch_size`** | `4` | **单次批大小**。每次训练时，模型不是一条一条读数据，而是把 4 个样本打包成一个大矩阵（Tensor）一起算，提升 GPU 效率。 |
#| **`shuffle`** | `True` | **是否打乱顺序**。每轮训练（Epoch）开始时，把数据的顺序随机洗牌。这能防止模型死记硬背数据的先后顺序，提高泛化能力。 |
#| **`drop_last`** | `True` | **是否丢弃尾数**。假设总共有 102 条数据，每 4 条打包一个 Batch，最后会剩 2 条。如果为 `True`，就把最后的 2 条扔掉，确保每个 Batch 的形状完全一致，避免报错。 |
#| **`num_workers`**| `0` | **多线程加载**。`0` 表示只用主线程。如果改成 `2` 或 `4`，PyTorch 会开启多个后台子进程并行去内存里抓数据，防止 CPU 供货太慢导致 GPU 闲置。 |
#向量列表”就是一个大列表，里面嵌套着许多小列表（每个小列表代表一句话）
#在 GPT 这类因果语言模型中，模型的目标是预测下一个词。所以，标签就是把输入文本整体往右移动一位的“标准答案”。
#张量标签列表”是指一个 Python 列表，里面塞满了一个接一个已经转换为 PyTorch Tensor 的标准答案。
################################################################################
with open("the-verdict.txt","r",encoding="utf-8") as f: 
    raw_text=f.read()
dataloader = create_dataloader_v1(
     raw_text,
     batch_size=1,
     max_length=4,
     stride=1,
     shuffle=False
)
data_iter=iter(dataloader)     #把 dataloader 变成一个“迭代器”（Iterator）
first_batch=next(data_iter)    #调用 next() 函数，从迭代器中获取下一个元素，并赋值给 first_batch
print(first_batch)
secode_batch=next(data_iter)
print(secode_batch)
##################################################################################
dataloader=create_dataloader_v1(raw_text,batch_size=8,max_length=4,stride=4,shuffle=False)

data_iter=iter(dataloader)
inputs,targets=next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
inputs,targets=next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)