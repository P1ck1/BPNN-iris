import torch
import torchvision.transforms as transforms
from PIL import Image
print("here0")
from LeNet5 import LeNet

if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print("here1")
    net2 = LeNet()
    print("here")
    net2.load_state_dict(torch.load('Lenet_66.7%.pth'))

    im = Image.open('1.jpg')
    im = transform(im)  # [C, H, W]
    # 输入pytorch网络中要求的格式是[batch，channel，height，width]，所以这里增加一个维度
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net2(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy() # 索引即classed中的类别
    print(classes[int(predict)])

    # 直接打印张量的预测结果
    with torch.no_grad():
        outputs = net2(im)
        predict = torch.softmax(outputs, dim=1) # [batch，channel，height，width],这里因为对batch不需要处理
    print(predict)
