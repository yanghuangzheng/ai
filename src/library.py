import urllib.request                                    # Rust 里的 reqwest 库
import re

url = ("https://raw.githubusercontent.com/rasbt/" "LLMs-from-scratch/main/ch02/01_main-chapter-code/" "the-verdict.txt")
file_path="the-verdict.txt"                              #声明下载到本地电脑后的文件名
urllib.request.urlretrieve(url,file_path)                #发起一个同步的 HTTP GET 请求，将远程服务器上的文件流（Stream）直接下载并写入到本地硬盘上，保存为 the-verdict.txt
with open("the-verdict.txt","r",encoding="utf-8") as f:  #这行语句在系统底层打开了一个指向 the-verdict.txt 的文件描述符（只读模式 "r"）。安全机制：with 关键字是 Python 的上下文管理器（Context Manager）。它在底层完美等同于 Rust 的 RAII 机制！当 with 缩进块内的代码执行完毕后，系统会自动调用 drop，强行关闭文件句柄并释放内存，绝对不会发生文件占用和内存泄漏。
    raw_text=f.read()
    print("Total number of character:",len(raw_text))
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words=sorted(set(preprocessed))
vocab = {token: integer for integer,token in enumerate(all_words)} 