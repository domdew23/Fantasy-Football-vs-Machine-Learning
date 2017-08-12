import re

text = '3.03'
text = text.split('.')
val = (int(text[0]) * 60) + int(text[1])
print(val)