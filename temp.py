import pickle
f = open('char_dict', 'rb')
dict_ori = pickle.load(f)# ‘New str to add’
dict_new = {value:key for key,value in dict_ori.items()}
print(dict_new)
str = ''
for i in range(300, 400):
    str += dict_new[i]
print(str)
f.close()



