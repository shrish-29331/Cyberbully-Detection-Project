import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from PIL import Image
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from models.Image_text_emotion_sentiment import Image_text_emotion_sentiment
from models.Memes_dataset import MemeDataset
# Import required libraries (if not already done)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import dataset_downloader.load_data as load_data


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

df = load_data()






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the dataset and dataloader
dataset = MemeDataset(df, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
model= Image_text_emotion_sentiment()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # Use DataParallel if multiple GPUs are available
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# Loss functions
loss_fn_sentiment = nn.CrossEntropyLoss().to(device)
loss_fn_emotion = nn.CrossEntropyLoss().to(device)
loss_fn_sarcasm = nn.BCEWithLogitsLoss().to(device)
loss_fn_bully = nn.CrossEntropyLoss().to(device)
loss_fn_harmful_score = nn.CrossEntropyLoss().to(device)
loss_fn_target = nn.CrossEntropyLoss().to(device)

# Training loop
for epoch in range(25):  # Set epochs accordingly
    model.train()

    total_loss = 0  # Initialize total loss for the epoch

    for images, text_input_ids, text_attention_mask, sentiment_labels, emotion_labels, sarcasm_labels, bully_labels, harmful_score_labels, target_labels in train_dataloader:
        # Move data to the GPU
    # for images, text_input_ids, text_attention_mask, bully_labels in train_dataloader:

        images = images.to(device)
        text_input_ids = text_input_ids.to(device)
        text_attention_mask = text_attention_mask.to(device)
        sentiment_labels = sentiment_labels.to(device)
        emotion_labels = emotion_labels.to(device)
        sarcasm_labels = sarcasm_labels.to(device)
        bully_labels = bully_labels.to(device)
        harmful_score_labels = harmful_score_labels.to(device)
        target_labels = target_labels.to(device)

        optimizer.zero_grad()  # Clear gradients at the start of each batch

        # Forward pass
        sentiment_out, emotion_out, sarcasm_out, bully_out, harmful_score_out, target_out = model(images, text_input_ids, text_attention_mask)

        # Compute loss for each task
        loss_sentiment = loss_fn_sentiment(sentiment_out, sentiment_labels)
        loss_emotion = loss_fn_emotion(emotion_out, emotion_labels)
        loss_sarcasm = loss_fn_sarcasm(sarcasm_out.squeeze(), sarcasm_labels.float())  # Squeeze if necessary
        loss_bully = loss_fn_bully(bully_out, bully_labels)
        loss_harmful_score = loss_fn_harmful_score(harmful_score_out, harmful_score_labels)
        loss_target = loss_fn_target(target_out, target_labels)

        # Total loss (sum or weigh the losses as needed)
        total_loss_batch = loss_sentiment + loss_emotion + loss_sarcasm + loss_bully + loss_harmful_score + loss_target

        # Backward pass and optimization
        total_loss_batch.backward()
        optimizer.step()  # Update model parameters

        total_loss += total_loss_batch.item()  # Accumulate loss for the epoch

    # Optionally clear cache at the end of each epoch
    torch.cuda.empty_cache()

    # Print the average loss for the epoch
    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')

    model.eval()  # Set model to evaluation mode
total_val_loss = 0

# Initialize lists to store true and predicted labels for each task
all_labels_bully = []
all_preds_bully = []

all_labels_sentiment = []
all_preds_sentiment = []

all_labels_emotion = []
all_preds_emotion = []

all_labels_sarcasm = []
all_preds_sarcasm = []

all_labels_harmful_score = []
all_preds_harmful_score = []

all_labels_target = []
all_preds_target = []

with torch.no_grad():  # Disable gradient calculation
    for images, text_input_ids, text_attention_mask, sentiment_labels, emotion_labels, sarcasm_labels, bully_labels, harmful_score_labels, target_labels in val_dataloader:
        # Move data to the GPU
        images = images.to(device)
        text_input_ids = text_input_ids.to(device)
        text_attention_mask = text_attention_mask.to(device)
        sentiment_labels = sentiment_labels.to(device)
        emotion_labels = emotion_labels.to(device)
        sarcasm_labels = sarcasm_labels.to(device)
        bully_labels = bully_labels.to(device)
        harmful_score_labels = harmful_score_labels.to(device)
        target_labels = target_labels.to(device)

        # Forward pass
        sentiment_out, emotion_out, sarcasm_out, bully_out, harmful_score_out, target_out = model(images, text_input_ids, text_attention_mask)

        # Compute loss for each task
        loss_sentiment = loss_fn_sentiment(sentiment_out, sentiment_labels)
        loss_emotion = loss_fn_emotion(emotion_out, emotion_labels)
        loss_sarcasm = loss_fn_sarcasm(sarcasm_out.squeeze(), sarcasm_labels.float())
        loss_bully = loss_fn_bully(bully_out, bully_labels)
        loss_harmful_score = loss_fn_harmful_score(harmful_score_out, harmful_score_labels)
        loss_target = loss_fn_target(target_out, target_labels)

        # Total loss
        total_val_loss += (loss_sentiment + loss_emotion + loss_sarcasm + loss_bully + loss_harmful_score + loss_target).item()

        # Get predictions for each task
        _, predicted_sentiment = torch.max(sentiment_out, 1)
        _, predicted_emotion = torch.max(emotion_out, 1)
        _, predicted_sarcasm = torch.max(sarcasm_out, 1)
        _, predicted_bully = torch.max(bully_out, 1)
        _, predicted_harmful_score = torch.max(harmful_score_out, 1)  # Assuming multi-class
        _, predicted_target = torch.max(target_out, 1)  # Assuming multi-class

        # Collect true and predicted labels for each task
        all_labels_sentiment.append(sentiment_labels.cpu().numpy())
        all_preds_sentiment.append(predicted_sentiment.cpu().numpy())

        all_labels_emotion.append(emotion_labels.cpu().numpy())
        all_preds_emotion.append(predicted_emotion.cpu().numpy())

        all_labels_sarcasm.append(sarcasm_labels.cpu().numpy())
        all_preds_sarcasm.append(predicted_sarcasm.cpu().numpy())

        all_labels_bully.append(bully_labels.cpu().numpy())
        all_preds_bully.append(predicted_bully.cpu().numpy())

        all_labels_harmful_score.append(harmful_score_labels.cpu().numpy())
        all_preds_harmful_score.append(predicted_harmful_score.cpu().numpy())

        all_labels_target.append(target_labels.cpu().numpy())
        all_preds_target.append(predicted_target.cpu().numpy())

avg_val_loss = total_val_loss / len(val_dataloader)

# Flatten lists for each task
all_labels_bully = np.concatenate(all_labels_bully)
all_preds_bully = np.concatenate(all_preds_bully)

all_labels_sentiment = np.concatenate(all_labels_sentiment)
all_preds_sentiment = np.concatenate(all_preds_sentiment)

all_labels_emotion = np.concatenate(all_labels_emotion)
all_preds_emotion = np.concatenate(all_preds_emotion)

all_labels_sarcasm = np.concatenate(all_labels_sarcasm)
all_preds_sarcasm = np.concatenate(all_preds_sarcasm)

all_labels_harmful_score = np.concatenate(all_labels_harmful_score)
all_preds_harmful_score = np.concatenate(all_preds_harmful_score)

all_labels_target = np.concatenate(all_labels_target)
all_preds_target = np.concatenate(all_preds_target)

# Calculate accuracy and F1 score for each task
accuracy_bully_SA_EM = accuracy_score(all_labels_bully, all_preds_bully)
f1_bully_SA_EM = f1_score(all_labels_bully, all_preds_bully, average='weighted')

accuracy_sentiment_SA_EM = accuracy_score(all_labels_sentiment, all_preds_sentiment)
f1_sentiment_SA_EM = f1_score(all_labels_sentiment, all_preds_sentiment, average='weighted')

accuracy_emotion_SA_EM = accuracy_score(all_labels_emotion, all_preds_emotion)
f1_emotion_SA_EM = f1_score(all_labels_emotion, all_preds_emotion, average='weighted')

accuracy_sarcasm_SA_EM = accuracy_score(all_labels_sarcasm, all_preds_sarcasm)
f1_sarcasm_SA_EM = f1_score(all_labels_sarcasm, all_preds_sarcasm, average='weighted')

accuracy_harmful_score_SA_EM = accuracy_score(all_labels_harmful_score, all_preds_harmful_score)
f1_harmful_score_SA_EM = f1_score(all_labels_harmful_score, all_preds_harmful_score, average='weighted')

accuracy_target_SA_EM = accuracy_score(all_labels_target, all_preds_target)
f1_target_SA_EM = f1_score(all_labels_target, all_preds_target, average='weighted')

print(f'Epoch {epoch}, Validation Loss: {avg_val_loss:.4f},\n'
      f'Bully Accuracy: {accuracy_bully_SA_EM:.4f}, F1 Score: {f1_bully_SA_EM:.4f},\n'
      f'Sentiment Accuracy: {accuracy_sentiment_SA_EM:.4f}, F1 Score: {f1_sentiment_SA_EM:.4f},\n'
      f'Emotion Accuracy: {accuracy_emotion_SA_EM:.4f}, F1 Score: {f1_emotion_SA_EM:.4f},\n'
      f'Sarcasm Accuracy: {accuracy_sarcasm_SA_EM:.4f}, F1 Score: {f1_sarcasm_SA_EM:.4f},\n'
      f'Harmful Score Accuracy: {accuracy_harmful_score_SA_EM:.4f}, F1 Score: {f1_harmful_score_SA_EM:.4f},\n'
      f'Target Accuracy: {accuracy_target_SA_EM:.4f}, F1 Score: {f1_target_SA_EM:.4f}')

