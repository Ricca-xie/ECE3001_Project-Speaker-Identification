import torch
import argparse
from tqdm import tqdm
import datetime
import time
from timm.utils import accuracy
from model.model import My_model
from audio_dataset import AudioDataset
from torch.utils.data import DataLoader

device = 'cuda:0'#'cpu'
# device = 'cpu'
device = torch.device(device)

def parse_config():
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--train_path', type=str, default='../stu_dataset/train')
    parser.add_argument('--test_path', type=str, default='../stu_dataset/test')
    parser.add_argument('--save_model_path', type=str, default='./model/checkpoints/vggbn/vggbn11')
    parser.add_argument('--load_model_path', type=str, default='./model/checkpoints/vggbn/vggbn11_epoch_19_stft.pt')

    # training parameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4,help="learning rate")
    parser.add_argument('--model_base', type=str, default="vggbn",help="model base: vgg, resnet, vggbn")

    # data processing parameters
    parser.add_argument("--max_len", default=800, type=int, help="max_len")
    parser.add_argument("--window_shift", default=256, type=int, help="hop shift")
    parser.add_argument("--window_length", default=510, type=int, help="window length") # 256
    parser.add_argument("--use_stft", default=True, type=bool, help="whether to use stft")
    return parser.parse_args()

def valid(args, model):
    print('Predcting...')
    audio_testset = AudioDataset(args.test_path, args.max_len, args.window_length, args.window_shift, args.use_stft)
    test_data = DataLoader(audio_testset, batch_size=args.batchsize, shuffle=False)

    model.eval()
    acc1_total = 0.
    acc5_total = 0.
    step = 0

    with torch.no_grad():
        for step, (x, label) in enumerate(tqdm(test_data)):
            x = x.to(dtype=torch.float32, device=device)
            label = label.to(device)
            result, pred = model(x)
            acc1, acc5 = accuracy(result, label.view(-1), topk=(1, 5))
            acc1, acc5 = acc1.item()/100, acc5.item()/100
            # loss_total += float(loss.item())
            acc1_total += acc1
            acc5_total += acc5
    print("Valid_acc1:{}, Valid_acc5: {}".format( acc1_total / (step+1), acc5_total / (step+1) ))
    return acc1_total / (step+1), acc5_total / (step+1)

def train(args):
    model = My_model(num_classes=92, model_base=args.model_base)

    # load pretrained model
    model.load_state_dict(torch.load(args.load_model_path,map_location=device))

    model = model.to(dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_epoch = -1
    best_acc1 = 0
    best_acc5 = 0
    best_model = 0

    # valid(args, model)
    audio_trainset = AudioDataset(args.train_path, args.max_len, args.window_length, args.window_shift, args.use_stft)
    print(f"Length of training set: {len(audio_trainset)}")
    train_data = DataLoader(audio_trainset, batch_size=args.batchsize, shuffle=True, drop_last=True)

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        acc1_total = 0.
        acc5_total = 0.
        loss_total = 0.
        for step, (x, label) in enumerate(tqdm(train_data)):
            x = x.to(dtype=torch.float32, device=device)
            label = label.to(device)
            optimizer.zero_grad()
            loss, result, pred = model(x, label)
            acc1, acc5 = accuracy(result, label.view(-1), topk=(1, 5))
            try:
                acc1, acc5 = acc1.item() / 100, acc5.item() / 100
            except:
                print("testt")
            loss.backward()
            optimizer.step()
            acc1_total += acc1
            acc5_total += acc5
            loss_total += float(loss.item())
            if step % args.print_every == 0 and step != 0:
                print('epoch %d, step %d, step_loss %.4f, step_acc1 %.4f, step_acc5 %.4f' % (epoch, step, loss_total/(step+1), acc1_total/(step+1), acc5_total/(step+1)))

        # save model
        # if epoch % args.save_every == 0 and epoch != 0:
        #     if args.use_stft:
        #         model_name = args.save_model_path+ "_epoch_"+ str(epoch)+'_stft.pt'
        #     else:
        #         model_name = args.save_model_path + "_epoch_"+ str(epoch) + 'no_stft.pt'
        #     torch.save(model.state_dict(), model_name)
        acc1, acc5 = valid(args, model)
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_acc5 = acc5
            best_epoch = epoch
            best_model = model
            # torch.save(model.state_dict(), args.checkpoint_path+'_pretrain.pt')
        print('best acc1 is: {}, acc5 is: {}, in epoch {}'.format(best_acc1, best_acc5, best_epoch))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = parse_config()
    train(args)
