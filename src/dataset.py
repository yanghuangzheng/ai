import urllib.request                                    # Rust 里的 reqwest 库
import re
import tiktoken

from importlib.metadata import version
from SimpleTokenizerV1 import SimpleTokenizerV1
from SimpleTokenizerV2 import SimpleTokenizerV2

#url = ("https://raw.githubusercontent.com/rasbt/" "LLMs-from-scratch/main/ch02/01_main-chapter-code/" "the-verdict.txt")
#file_path="the-verdict.txt"                              #声明下载到本地电脑后的文件名
#urllib.request.urlretrieve(url,file_path)                #发起一个同步的 HTTP GET 请求，将远程服务器上的文件流（Stream）直接下载并写入到本地硬盘上，保存为 the-verdict.txt
with open("the-verdict.txt","r",encoding="utf-8") as f:  #这行语句在系统底层打开了一个指向 the-verdict.txt 的文件描述符（只读模式 "r"）。安全机制：with 关键字是 Python 的上下文管理器（Context Manager）。它在底层完美等同于 Rust 的 RAII 机制！当 with 缩进块内的代码执行完毕后，系统会自动调用 drop，强行关闭文件句柄并释放内存，绝对不会发生文件占用和内存泄漏。
    raw_text=f.read()
    #print("Total number of character:",len(raw_text))
    #print(raw_text[:99])
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    #print(len(preprocessed))
    #print(preprocessed[:30])
#退出缩进代码结束f关闭
all_words=sorted(set(preprocessed))  #先转换成列表去重然后再用set扔进红黑树中排序
all_words.extend(["<|unk|>",""])
voca_size=len(all_words)
#print(voca_size)

vocab = {token: integer for integer,token in enumerate(all_words)}  #enumerate(all_words) ➔ 给排好队的单词发“排队号”  第一轮：吐出 (0, "brown") ➔ 赋值给前面的变量 (integer, token)  第二轮：吐出 (1, "dog")  然后转换变成文字为键 
#print(len(vocab.items()))
#for i,item in enumerate(vocab.items()):
#    print(item)
#    if i>=50:
#       break

#tokenizer = SimpleTokenizerV1(vocab)
#text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.  """
#ids=tokenizer.encode(text)
#print(ids)
#print(tokenizer.decode(ids))
#text = "Hello, do you like tea?"
#print(tokenizer.encode(text))

#for i,item in enumerate(list(vocab.items())[-5:]):
#    print(item)
#text1="Hello , do you like tea?"
#text2="in the sunlit terraces of the palace"
#text=" ".join((text1,text2))
#print(text)
#tokenizer=SimpleTokenizerV2(vocab)
#print(tokenizer.decode(tokenizer.encode(text)))
print("tiktoken版本",version("tiktoken"))
tokenizer=tiktoken.get_encoding("gpt2")#类似创造了 SimpleTokenizerV系列
text=(
    "Hello,do you like tea? "
    "In the sunlit terraces"
    "of someunknownPlace."
)
integers = tokenizer.encode(text,allowed_special={" "}) 
print(integers)
strings = tokenizer.decode(integers)
print(strings)



