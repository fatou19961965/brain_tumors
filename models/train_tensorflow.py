import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class TrainerTF:
    def __init__(self, model, train_dataset, test_dataset, lr, wd, epochs):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=wd)
        self.train_loss_results = []
        self.train_accuracy_results = []
        self.train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    def train(self, save=False, plot=False):
        for epoch in range(self.epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            self.train_accuracy_metric.reset_state()
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for x_batch, y_batch in tqdm(self.train_dataset, leave=False):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch, training=True)
                    loss_value = self.loss_fn(y_batch, logits)
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                epoch_loss_avg.update_state(loss_value)
                self.train_accuracy_metric.update_state(y_batch, logits)
            train_loss = epoch_loss_avg.result().numpy()
            train_acc = self.train_accuracy_metric.result().numpy() * 100
            self.train_loss_results.append(train_loss)
            self.train_accuracy_results.append(train_acc)
            print(f" → Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        if save:
            self.model.save(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'thierno_model.tensorflow.keras'))
        if plot:
            self.plot_training_history()

    def evaluate(self):
        self.test_accuracy_metric.reset_state()
        loss_metric = tf.keras.metrics.Mean()
        for x_batch, y_batch in tqdm(self.test_dataset, desc="Evaluating", leave=False):
            logits = self.model(x_batch, training=False)
            loss = self.loss_fn(y_batch, logits)
            loss_metric.update_state(loss)
            self.test_accuracy_metric.update_state(y_batch, logits)
        test_loss = loss_metric.result().numpy()
        test_acc = self.test_accuracy_metric.result().numpy() * 100
        print(f"\nTest Accuracy: {test_acc:.2f}%  |  Test Loss: {test_loss:.4f}")
        return test_acc, test_loss

    def plot_training_history(self):
        epochs_range = range(1, len(self.train_loss_results) + 1)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(epochs_range, self.train_loss_results, color=color_loss, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)
        ax2 = ax1.twinx()
        color_acc = 'tab:red'
        ax2.set_ylabel('Accuracy (%)', color=color_acc)
        ax2.plot(epochs_range, self.train_accuracy_results, color=color_acc, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)
        plt.title('Training Loss and Accuracy')
        fig.tight_layout()
        plt.savefig('training_history_tensorflow.png')