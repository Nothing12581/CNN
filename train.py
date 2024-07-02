import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
from model import MobileNetV2
import os
import time


if __name__ == '__main__':
    # device : GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 数据转换
    data_transform = {
        "train": transforms.Compose([transforms.Grayscale(num_output_channels=1),  # 转为灰度图像
                                     transforms.Resize((32, 32)),
                                     transforms.ToTensor()]),  # 初始化
        "val": transforms.Compose([transforms.Grayscale(num_output_channels=1),  # 转为灰度图像
                                   transforms.Resize((32, 32)),
                                     transforms.ToTensor()]) }

    data_root = r"W:\desktop\work\data" # 获取数据集根目录
    train_image_path = data_root + "/train/"  # 训练数据集目录
    train_dataset = datasets.ImageFolder(root=train_image_path,
                                         transform=data_transform["train"])

    print("训练数据集大小:", train_dataset)

    batch_size = 32
    train_loader = DataLoader(train_dataset,  # 加载训练集数据
                              batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)


    val_image_path = data_root + "/test/"  # 训练数据集目录
    validate_dataset = datasets.ImageFolder(root=val_image_path,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = DataLoader(validate_dataset,  # 加载验证集数据
                                 batch_size=batch_size, shuffle=True,
                                 num_workers=0, pin_memory=True)

    test_data_iter = iter(validate_loader)
    test_image, test_label = next(test_data_iter)

    net = MobileNetV2()  # 设置分类数为41，初始化权重
    net.to(device)
    # 损失函数:这里用交叉熵
    loss_function = nn.CrossEntropyLoss()
    # 优化器 这里用Adam
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # 训练参数保存路径
    save_path = './MobileNetV2.pth' #只要是pth文件就可以
    # 训练过程中最高准确率
    best_acc = 0.0

    # 开始进行训练和测试，训练一轮，测试一轮
    for epoch in range(50): # 替换成任意正整数
        # train
        net.train()  # 训练过程中，使用之前定义网络中的dropout
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader):
            images, labels = data  # 提取图像数据和标签
            optimizer.zero_grad()  # 清空之前的梯度信息
            outputs = net(images.to(device))  # 将数据存放到设备
            loss = loss_function(outputs, labels.to(device))  # 计算损失值
            loss.backward()  # 损失后向传播到每个神经元
            optimizer.step()  # 更新每个神经元的参数
            running_loss += loss.item()  # 累加损失
            # 打印训练进度
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print(time.perf_counter() - t1)

        # validate
        net.eval()  # 测试过程中不需要dropout，使用所有的神经元
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():  # 进行验证，并不进行梯度跟踪
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # 得到预测结果
                acc += (predict_y == val_labels.to(device)).sum().item()  # 累计预测准确率
            val_accurate = acc / val_num  # 求得测试集准确率
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)  # 保存模型
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, running_loss / len(train_loader), val_accurate))
    print('Finished Training')

