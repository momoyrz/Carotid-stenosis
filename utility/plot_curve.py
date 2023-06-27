import matplotlib.pyplot as plt

# 读取文本文件，将每一行的内容解析为字典
data = []
log_dir = '../results/log.txt'
with open(log_dir, 'r') as file:
    for line in file:
        entry = eval(line.replace("Infinity", "float('inf')"))
        data.append(entry)

# 提取训练和测试的loss值
train_loss = [entry['train_loss'] for entry in data]
test_loss = [entry['test_loss'] for entry in data]
test_acc = [entry['test_acc1'] for entry in data]
# 绘制loss曲线
epochs = range(len(data))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, test_loss, label='Test Loss')
# plt.plot(epochs, test_acc, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()