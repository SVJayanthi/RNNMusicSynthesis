import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def load_histories(model_save_dir):
    history_path = os.path.join(model_save_dir, "history.json")
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def plot_model_histories(histories, save_directory):
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plotting loss
    for model_name, history in histories.items():
        epochs = range(1, len(history['loss']) + 1)
        axes[0].plot(epochs, history['loss'], label=f'{model_name} - Train Loss')
        axes[0].plot(epochs, history['val_loss'], label=f'{model_name} - Val Loss', linestyle='--')
    
    axes[0].set_title('Model Losses')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Plotting accuracy
    for model_name, history in histories.items():
        epochs = range(1, len(history['sparse_categorical_accuracy']) + 1)
        axes[1].plot(epochs, history['sparse_categorical_accuracy'], label=f'{model_name} - Train Accuracy')
        axes[1].plot(epochs, history['val_sparse_categorical_accuracy'], label=f'{model_name} - Val Accuracy', linestyle='--')
    
    axes[1].set_title('Model Accuracies')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    
    save_fig_path = os.path.join(save_directory, "loss_accuracy_curve.png")
    plt.savefig(save_fig_path)

def plot_test_accuracies(histories, save_directory):
    # Names of the models
    model_names = list(histories.keys())
    # Test accuracy values
    accuracies = list( [h["test_accuracy"]  for h in histories.values() ])
    
    # Creating the bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(model_names, accuracies, color='dodgerblue', alpha=0.7)
    
    # Adding titles and labels
    plt.title('Comparison of Test Accuracies Across Models')
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    
    # Optionally, add the accuracy values on top of the bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    
    # Show the plot
    # plt.ylim(0.7, 0.9)  # Adjust y-axis limits to make the chart clearer
    
    save_fig_path = os.path.join(save_directory, "test_accuracy.png")
    plt.savefig(save_fig_path)


if __name__ == '__main__':
    save_plots_dir = ".\save_plots"
    now = datetime.now()
    datetime_str = now.strftime("%y_%m_%d_%H_%M_%S")
    save_directory = os.path.join(save_plots_dir, datetime_str)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    dictionary_model_saves  = {
        "gru": "training_checkpoints/gru/24_06_01_15_06_54",
        "lstm": "training_checkpoints/lstm/24_06_01_16_35_24"
    }
    model_saves_path = os.path.join(save_directory, "dictionary_model_saves.json")
    with open(model_saves_path, 'w') as f:
        json.dump(dictionary_model_saves, f)

    dictionary_model_histories = {}
    for key in dictionary_model_saves:
        dictionary_model_histories[key] = load_histories(dictionary_model_saves[key])
        
    plot_model_histories(dictionary_model_histories, save_directory)
    
    
    plot_test_accuracies(dictionary_model_histories, save_directory)
    
