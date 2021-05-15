import torch
from PIL import Image
from train_val import val_transforms,classes
import matplotlib.pyplot as plt

def predict(img_path):
    net=torch.load('./models/model.pth')
    net=net.cuda()
    torch.no_grad()
    img=Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    img=val_transforms(img).unsqueeze(0)
    img_ = img.cuda()
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    plt.title('This picture maybe : '+classes[predicted[0]])
    plt.show()
    #print('This picture maybe :',classes[predicted[0]])
if __name__ == '__main__':
    predict('./test/image(100).jpg')