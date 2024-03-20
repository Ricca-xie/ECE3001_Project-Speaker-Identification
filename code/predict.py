import numpy as np
import torch
import argparse
from model.model import My_model
from tqdm import tqdm
from timm.utils import accuracy
from audio_dataset import AudioDataset
from torch.utils.data import DataLoader

device = 'cuda:0'
# device = 'cpu'
device = torch.device(device)

def parse_config():
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--train_path', type=str, default='../stu_dataset/train')
    parser.add_argument('--test_path', type=str, default='../stu_dataset/test')
    parser.add_argument('--save_model_path', type=str, default='./model/checkpoints/vgg/vgg11')
    parser.add_argument('--load_model_path', type=str, default='./model/checkpoints/vgg/vgg11_epoch_10_stft.pt')

    # training parameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4,help="learning rate")
    parser.add_argument('--model_base', type=str, default="vgg",help="model base: vgg, resnet, vggbn")

    # data processing parameters
    parser.add_argument("--max_len", default=800, type=int, help="max_len")
    parser.add_argument("--window_shift", default=256, type=int, help="hop shift")
    parser.add_argument("--window_length", default=510, type=int, help="window length") # 256
    parser.add_argument("--use_stft", default=True, type=bool, help="whether to use stft")
    return parser.parse_args()

def predict(args):
    audio_testset = AudioDataset(args.test_path, args.max_len, args.window_length, args.window_shift, args.use_stft)
    test_data = DataLoader(audio_testset, batch_size=args.batchsize, shuffle=False)

    model = My_model(num_classes=92, model_base=args.model_base)
    model.load_state_dict(torch.load(args.load_model_path,map_location=device))
    model = model.to(dtype=torch.float32, device=device)
    model.eval()
    loss_total = 0.
    acc1_total = 0.
    acc5_total = 0.

    with torch.no_grad():

        for step, (x, label) in enumerate(test_data):
            x = x.to(dtype=torch.float32, device=device)
            label = label.to(device)
            result, pred = model(x)
            acc1, acc5 = accuracy(result, label.view(-1), topk=(1, 5))
            acc1, acc5 = acc1.item()/100, acc5.item()/100
            acc1_total += acc1
            acc5_total += acc5

    print("Test_acc1:{}, Test_acc5: {}".format(acc1_total / (step + 1), acc5_total / (step + 1)))
    # return acc1_total / (step + 1), acc5_total / (step + 1)


if __name__ == "__main__":
    args = parse_config()
    predict(args)
