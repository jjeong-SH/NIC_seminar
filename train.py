import os
import pickle
import argparse
import time
import json
import nltk
import random
import torch
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from vocabulary import Vocabulary
from dataset import get_loader
from model import EncoderCNN, DecoderRNN

# image directory
train_image_dir = '/flickr30k_images/resized_train/images'
val_image_dir = '/flickr30k_images/resized_val/images'
test_image_dir = '/flickr30k_images/resized_test/images'

# caption directory
train_caption_path = '/flickr30k_images/resized_train/captions.txt'
val_caption_path = '/flickr30k_images/resized_val/captions.txt'
test_caption_path = '/flickr30k_images/resized_test/captions.txt'


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True


def train(args):
    seed_everything(args.seed)  # def: 123

    # -- settings
    use_cuda = torch.cuda.is_available()
    print("USE CUDA> ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    model_path = "/trained_models/"
    vocab_path = "vocab.pkl"
    log_path = '/log/'
    crop_size = 224

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # 사전 학습된(pre-trained) ResNet에 적용된 전처리 및 정규화 파라미터를 그대로 사용합니다.
    train_transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    val_transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # 데이터 로더(data loader) 선언
    train_data_loader = get_loader(train_image_dir, train_caption_path, vocab, train_transform, batch_size=args.batch_size,
                                   shuffle=True, num_workers=4, testing=False)
    val_data_loader = get_loader(val_image_dir, val_caption_path, vocab, val_transform, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, testing=False)
    test_data_loader = get_loader(test_image_dir, test_caption_path, vocab, test_transform, batch_size=args.batch_size, shuffle=False,
                                  num_workers=4, testing=True)

    # 모델 하이퍼 파라미터 설정
    embed_size = 256  # 임베딩(embedding) 차원
    hidden_size = 512  # LSTM hidden states 차원
    num_layers = 1  # LSTM의 레이어 개수

    # 모델 객체 선언
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

    num_epochs = args.epochs
    learning_rate = 0.001

    log_step = 50  # 로그를 출력할 스텝(step)
    save_step = 1000  # 학습된 모델을 저장할 스텝(step)

    # 손실(loss) 및 최적화 함수 선언
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    # logging
    logger = SummaryWriter(log_dir=log_path)
    with open(os.path.join(log_path, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    ## --train
    start_time = time.time()  # 전체 학습 시간 측정

    # 모델 학습 진행
    for epoch in range(num_epochs):

        # 먼저 학습 진행하기
        print("[ Training ]")
        total_loss = 0
        total_count = 0
        total_step = len(train_data_loader)
        for i, (images, captions, lengths) in enumerate(train_data_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # 순전파(forward), 역전파(backward) 및 학습 진행
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # 손실(loss) 값 계산
            total_loss += loss.item()
            total_count += images.shape[0]

            # 로그(log) 정보 출력
            if i % log_step == 0:
                avg_train_loss = total_loss / total_count
                print('Epoch [{}/{}], Step [{}/{}], Average Loss: {:.4f}, Perplexity: {:5.4f}, Elapsed time: {:.4f}s'
                      .format(epoch, num_epochs, i, total_step, avg_train_loss, np.exp(loss.item()),
                              time.time() - start_time))
                logger.add_scaler("Train/loss", avg_train_loss, epoch * len(train_data_loader) + i)
        scheduler.step(avg_train_loss)

        # 모델 파일 저장하기
        torch.save(decoder.state_dict(), os.path.join(model_path, f'decoder-{epoch + 1}.ckpt'))
        torch.save(encoder.state_dict(), os.path.join(model_path, f'encoder-{epoch + 1}.ckpt'))
        print(f"Model saved: {os.path.join(model_path, f'decoder-{epoch + 1}.ckpt')}")
        print(f"Model saved: {os.path.join(model_path, f'encoder-{epoch + 1}.ckpt')}")

        # 학습 이후에 평가 진행하기
        print("[ Validation ]")
        total_loss = 0
        total_count = 0
        total_step = len(val_data_loader)
        with torch.no_grad():
            for i, (images, captions, lengths) in enumerate(val_data_loader):
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # 순전파(forward) 진행
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                loss = criterion(outputs, targets)

                # 손실(loss) 값 계산
                total_loss += loss.item()
                total_count += images.shape[0]

                # 로그(log) 정보 출력
                if i % log_step == 0:
                    avg_val_loss = total_loss / total_count
                    print(
                        'Epoch [{}/{}], Step [{}/{}], Average Loss: {:.4f}, Perplexity: {:5.4f}, Elapsed time: {:.4f}s'
                        .format(epoch, num_epochs, i, total_step, avg_val_loss, np.exp(loss.item()),
                                time.time() - start_time))
                    logger.add_scaler("Val/loss", avg_val_loss, epoch * len(val_data_loader) + i)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128, help='choose batch size (default: 128)')
    args = parser.parse_args()

    train(args)

