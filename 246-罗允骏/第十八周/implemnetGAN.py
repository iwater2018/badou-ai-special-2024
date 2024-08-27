import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm  import tqdm
import matplotlib.pyplot as plt

class Generate_Model(torch.nn.Module):
    '''
    生成器
    '''
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Sequential(
            torch.nn.Linear(in_features=128,out_features=256),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=256,out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512,out_features=784),
            torch.nn.Tanh()
        )
    def forward(self,x):
        x=self.fc(x)
        return x

class Distinguish_Model(torch.nn.Module):
    '''
    判别器
    '''
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Sequential(
            torch.nn.Linear(in_features=784,out_features=512),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=512,out_features=256),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=256,out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128,out_features=1),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        x=self.fc(x)
        return x
def train():
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #判断是否存在可用GPU
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ]) #图片标准化
    train_data = MNIST("./data", transform=transformer,download=True) #载入图片
    dataloader = DataLoader(train_data, batch_size=64,num_workers=4, shuffle=True) #将图片放入数据加载器

    D = Distinguish_Model().to(device) #实例化判别器
    G = Generate_Model().to(device) #实例化生成器

    D_optim = torch.optim.Adam(D.parameters(), lr=1e-4) #为判别器设置优化器
    G_optim = torch.optim.Adam(G.parameters(), lr=1e-4) #为生成器设置优化器

    loss_fn = torch.nn.BCELoss() #损失函数

    epochs = 100 #迭代100次
    for epoch in range(epochs):
        dis_loss_all=0 #记录判别器损失损失
        gen_loss_all=0 #记录生成器损失
        loader_len=len(dataloader) #数据加载器长度
        for step,data in tqdm(enumerate(dataloader), desc="第{}轮".format(epoch),total=loader_len):
            # 先计算判别器损失
            sample,label=data #获取样本，舍弃标签
            sample = sample.reshape(-1, 784).to(device) #重塑图片
            sample_shape = sample.shape[0] #获取批次数量
            #从正态分布中抽样
            sample_z = torch.normal(0, 1, size=(sample_shape, 128),device=device)

            Dis_true = D(sample) #判别器判别真样本

            true_loss = loss_fn(Dis_true, torch.ones_like(Dis_true)) #计算损失

            fake_sample = G(sample_z) #生成器通过正态分布抽样生成数据
            Dis_fake = D(fake_sample.detach()) #判别器判别伪样本
            fake_loss = loss_fn(Dis_fake, torch.zeros_like(Dis_fake)) #计算损失

            Dis_loss = true_loss + fake_loss #真假加起来
            D_optim.zero_grad()
            Dis_loss.backward() #反向传播
            D_optim.step()

            # 生成器损失
            Dis_G = D(fake_sample) #判别器判别
            G_loss = loss_fn(Dis_G, torch.ones_like(Dis_G)) #计算损失
            G_optim.zero_grad()
            G_loss.backward() #反向传播
            G_optim.step()
            with torch.no_grad():
                dis_loss_all+=Dis_loss #判别器累加损失
                gen_loss_all+=G_loss #生成器累加损失
        with torch.no_grad():
            dis_loss_all=dis_loss_all/loader_len
            gen_loss_all=gen_loss_all/loader_len
            print("判别器损失为：{}".format(dis_loss_all))
            print("生成器损失为：{}".format(gen_loss_all))
        torch.save(G, "./model/G.pth") #保存模型
        torch.save(D, "./model/D.pth") #保存模型
if __name__ == '__main__':
    # train() #训练模型
    model_G=torch.load("./model/G.pth",map_location=torch.device("cpu")) #载入模型
    fake_z=torch.normal(0,1,size=(10,128))  #抽样数据
    result=model_G(fake_z).reshape(-1,28,28)  #生成数据
    result=result.detach().numpy()

    #绘制
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(result[i])
        plt.gray()
    plt.show()
