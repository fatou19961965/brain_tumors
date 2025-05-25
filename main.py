import torch
import argparse
import os
import sys
from models.prep_pytorch import get_data
from models.cnn_pytorch import get_pytorch_model
from models.cnn_tensorflow import get_tensorflow_model
from models.train_pytorch import Trainer
from models.train_tensorflow import TrainerTF
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a CNN model")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0001, help="Weight decay")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help="Mode: train or eval")
    parser.add_argument('--framework', type=str, choices=['pytorch', 'tensorflow'], default='pytorch', help="Framework: pytorch or tensorflow")
    parser.add_argument('--cuda', action='store_true', help="Use GPU if available")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    if args.framework == 'pytorch':
        train_dataloader, test_dataloader = get_data()
        model = get_pytorch_model().to(device)
        if args.mode == 'eval':
            model_path = os.path.join('models', 'thierno_model.torch')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                print(f"Model not found at: {model_path}")
                return
        trainer = Trainer(model, train_dataloader, test_dataloader, args.lr, args.wd, args.epochs, device)
        if args.mode == 'train':
            trainer.train(save=True, plot=True)
        trainer.evaluate()
    else:  # tensorflow
        train_dataloader, test_dataloader = get_data()
        # Convert PyTorch dataloaders to TensorFlow datasets
        def gen_dataset(dataloader):
            for images, labels in dataloader:
                yield images.numpy().transpose(0, 2, 3, 1), labels.numpy()
        train_dataset = tf.data.Dataset.from_generator(
            lambda: gen_dataset(train_dataloader),
            output_types=(tf.float32, tf.int64),
            output_shapes=([None, 224, 224, 3], [None])
        ).batch(64)
        test_dataset = tf.data.Dataset.from_generator(
            lambda: gen_dataset(test_dataloader),
            output_types=(tf.float32, tf.int64),
            output_shapes=([None, 224, 224, 3], [None])
        ).batch(64)
        model = get_tensorflow_model()
        if args.mode == 'eval':
            model_path = os.path.join('models', 'thierno_model.tensorflow.keras')
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
            else:
                print(f"Model not found at: {model_path}")
                return
        trainer = TrainerTF(model, train_dataset, test_dataset, args.lr, args.wd, args.epochs)
        if args.mode == 'train':
            trainer.train(save=True, plot=True)
        trainer.evaluate()

if __name__ == '__main__':
    main()