import re
text = "Hello, world. this, is a test"
#result = re.split(r'(\s)',text) 
#result = re.split('(\s)',text) 
result = re.split(r'([,.]|\s)', text)
result = [item for item in result if item.strip()]  #strip() 的作用是脱掉字符串两端的空白字符（包括空格、换行、制表符） 任何空的东西（比如空字符串 ''）在 if 面前都会被自动判定为 False！
print(result)