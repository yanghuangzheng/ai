import tiktoken


from importlib.metadata import version



print("tiktoken版本",version("tiktoken"))
tokenizer=tiktoken.get_encoding("gpt2")
text=(
    "Hello,do you like tea? "
    "In the sunlit terraces"
    "of someunknownPlace."
)
integers = tokenizer.encode(text,allowed_special={" "})#调用 tiktoken 编译器，把文本变量 text 切碎并加密成一串大模型专用的数字 ID（整数列表），同时强制规定：如果在文本里看到了空格 " "，不要报错，把它当成合法的特殊标记放行处理。”
print(integers)
strings = tokenizer.decode(integers)
print(strings)
text1=("AKwirw ier")
integers = tokenizer.encode(text1,allowed_special={" "})
print(integers)
strings = tokenizer.decode(integers)
print(strings)
######################################################################### SlidingWindow
with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()
    enc_text=tokenizer.encode(raw_text)   #单词对应的数字 这里的分词和书上不一样
    print(len(enc_text))
context_size=4
x=enc_text[:context_size]                 #前4个
y=enc_text[1:context_size+1]              #后退一位
print(f"x:{x}")
print(f"y:{y}")

for i in range(1,context_size+1):
    context=enc_text[:i]
    desired=enc_text[i]
    print(context,"---->",desired)        #箭头（—->）左边的所有内容指的是LLM接收到的输入   箭头右边的标记ID表示LLM需要预测的目标标记ID
    # 让我们重复之前的代码，但将标记ID转换为文本：
for i in range(1,context_size+1):
    context=enc_text[:i]
    desired=enc_text[i]
    print(tokenizer.decode(context),"---->",tokenizer.decode([desired]))   