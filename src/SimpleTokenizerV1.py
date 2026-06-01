import re
 #实现简单的编码和解码器
class SimpleTokenizerV1:                                                #struct         
    def __init__ (self,vocab):                                             #new
     self.str_to_int=vocab                                              #在结构体上挂载一个str_to_int
     self.int_to_str={i:s for s,i in vocab.items()}                     #颠倒
     
    def encode(self,text):                                              #字母换成数字                                 
       preprocessed=re.split(r'([,.?_!"()\']|--|\s)', text)
       preprocessed=[
          item.strip() for item in preprocessed if item.strip()
       ]
       ids = [self.str_to_int[s] for s in preprocessed]
       return ids
    

    def decode(self,ids):                                              #字母变成字母串
       text=" ".join([self.int_to_str[i] for i in ids])                 #先把数字数组查反向词表换回单词数组，然后用“空格”把它们连成一长串字符串
       text=re.sub(r'\s+([,.?!"()\'])',r'\1', text)                     #因为上面用空格拼接时，标点符号前面会多出一个多余的空格（比如变成 "Hello , world ."）。这行正则的作用是“自动把标点符号前面的空格删掉”，优雅地还原成 "Hello, world."。
       return text


