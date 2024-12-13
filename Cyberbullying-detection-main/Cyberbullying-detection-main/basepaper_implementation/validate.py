# Import required libraries (if not already done)
import sys
sys.path.append("..")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from MemeDatasetClipVGG import MemeDatasetClipVGG
from MemeModelCLIPVGG import MemeModelCLIPVGG
from dataset_downloader.load_data import load_data
from torch.utils.data import TensorDataset, DataLoader, random_split
import dataset_downloader.load_data as load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import pretrained_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, vgg19, roberta_model, tokenizer, transform, device = pretrained_models.get_models()


# Make the Loss Functions
# Loss functions
loss_fn_sentiment = nn.CrossEntropyLoss().to(device)
loss_fn_emotion = nn.CrossEntropyLoss().to(device)
loss_fn_sarcasm = nn.BCEWithLogitsLoss().to(device)
loss_fn_bully = nn.CrossEntropyLoss().to(device)
loss_fn_harmful_score = nn.CrossEntropyLoss().to(device)
loss_fn_target = nn.CrossEntropyLoss().to(device)


def validate_model(model, val_dataloader, epoch=15):

    # Validation
    model.eval()
    val_running_loss = 0.0

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

    epoch = 20

    with torch.no_grad():
        torch.cuda.empty_cache()
        batch_iterator = tqdm(val_dataloader, desc=f"Processing Validation {epoch:02d}")
        for data_sample in batch_iterator:
            #extract features
            image = data_sample['image'].to(device)
            image_clip_input = data_sample['image_clip_input'].to(device)
            image_vgg_feature = data_sample['image_vgg_feature'].to(device)
            text_clip_input = data_sample['text_clip_input'].to(device)
            text_roberta_embedding = data_sample['text_roberta_embedding'].to(device)

            #extract lables
            sentiment_labels = data_sample['sentiment'].to(device)
            emotion_labels = data_sample['emotion'].to(device)
            sarcasm_labels = data_sample['sarcasm'].to(device)
            bully_labels = data_sample['bully_label'].to(device)
            harmful_score_labels = data_sample['harmful-score'].to(device)
            target_labels = data_sample['target'].to(device)
            
            # Forward Pass
            sentiment_output, emotion_output, sarcasm_output, bully_output, harmful_score_output, target_output = model(image, image_clip_input, image_vgg_feature, text_clip_input, text_roberta_embedding )

            # Calculate Loss
            # Compute the losses
            loss_sentiment = loss_fn_sentiment(sentiment_output, sentiment_labels)
            loss_emotion = loss_fn_emotion(emotion_output, emotion_labels)
            loss_sarcasm = loss_fn_sarcasm(sarcasm_output.squeeze(), sarcasm_labels.float())
            loss_bully = loss_fn_bully(bully_output, bully_labels)
            loss_harmful_score = loss_fn_harmful_score(harmful_score_output, harmful_score_labels)
            loss_target = loss_fn_target(target_output, target_labels)
            
            # Combine the losses
            loss = loss_sentiment + loss_emotion + loss_sarcasm + loss_bully + loss_harmful_score + loss_target
            
            # Update running loss
            val_running_loss += loss.item()

            # Get predictions for each task
            _, predicted_sentiment = torch.max(sentiment_output, 1)
            _, predicted_emotion = torch.max(emotion_output, 1)
            _, predicted_sarcasm = torch.max(sarcasm_output, 1)
            _, predicted_bully = torch.max(bully_output, 1)
            _, predicted_harmful_score = torch.max(harmful_score_output, 1)  # Assuming multi-class
            _, predicted_target = torch.max(target_output, 1)  # Assuming multi-class

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
            
        # Print the average loss for this epoch
        val_running_loss = val_running_loss / len(train_dataloader)
        vals.append(val_running_loss)
        print(f'Epoch 20, Val_Loss: {val_running_loss:.4f}')


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

    print(f'Epoch {epoch}, Validation Loss: {val_running_loss:.4f},\n'
        f'Bully Accuracy: {accuracy_bully_SA_EM:.4f}, F1 Score: {f1_bully_SA_EM:.4f},\n'
        f'Sentiment Accuracy: {accuracy_sentiment_SA_EM:.4f}, F1 Score: {f1_sentiment_SA_EM:.4f},\n'
        f'Emotion Accuracy: {accuracy_emotion_SA_EM:.4f}, F1 Score: {f1_emotion_SA_EM:.4f},\n'
        f'Sarcasm Accuracy: {accuracy_sarcasm_SA_EM:.4f}, F1 Score: {f1_sarcasm_SA_EM:.4f},\n'
        f'Harmful Score Accuracy: {accuracy_harmful_score_SA_EM:.4f}, F1 Score: {f1_harmful_score_SA_EM:.4f},\n'
        f'Target Accuracy: {accuracy_target_SA_EM:.4f}, F1 Score: {f1_target_SA_EM:.4f}')


def main():
    df = load_data()

    # Load the dataset (assume the dataframe and image directory are available)
    dataset_clip_vgg = MemeDatasetClipVGG(df, transform, clip_model, vgg19)

    train_size = int(0.8 * len(dataset_clip_vgg))
    val_size = len(dataset_clip_vgg) - train_size

    # Split Dataset
    train_dataset, val_dataset = random_split(dataset_clip_vgg, [train_size, val_size])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Initialize the model and move it to the GPU

    img_clip_inp_dim = 512
    vgg_inp_dim = 25088
    text_clip_inp_dim = 512
    text_roberta_inp_dim = 768

    model = MemeModelCLIPVGG(img_clip_inp_dim, vgg_inp_dim, text_clip_inp_dim, text_roberta_inp_dim)
    # Initialize the Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.load_state_dict(torch.load("model_chekpoints/checkpoint_with_model.pth", weights_only=True))
    model.to(device) 

    validate_model(model, val_dataloader, epoch=15)