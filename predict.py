import torch
from model import MobileNetV2
from PIL import Image
from torchvision import transforms
import pickle

def get_predict():
    data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # 转为灰度图像
                                     transforms.Resize((32, 32)),
                                     transforms.ToTensor()])  # 初始化
    img = Image.open("test.png") # 加载图片，自定义的图片名称
    img = data_transform(img) # 图片转换为矩阵
    # 对数据维度进行扩充
    img = torch.unsqueeze(img, dim=0)
    # 创建模型
    model = MobileNetV2()
    # 加载模型权重
    model_weight_path = "./MobileNetV2.pth" #与train.py里的文件名对应
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img)) # 图片压缩
        predict = torch.softmax(output, dim=0) # 求softmax值
        predict_cla = torch.argmax(predict).numpy() # 预测分类结果
        f = open('char_dict', 'rb')
        dict_ori = pickle.load(f)  # ‘New str to add’
        dict_new = {value: key for key, value in dict_ori.items()}
        print("预测结果为",dict_new[predict_cla + 300])
        return dict_new[predict_cla + 300]

