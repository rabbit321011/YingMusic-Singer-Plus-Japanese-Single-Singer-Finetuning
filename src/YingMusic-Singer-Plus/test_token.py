from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer
t = CNENTokenizer()
print("CN:", t.encode("你好世界"))
print("EN:", t.encode("hello"))

















