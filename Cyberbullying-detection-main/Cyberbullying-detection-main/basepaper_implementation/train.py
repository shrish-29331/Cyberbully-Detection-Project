import sys
sys.path.append("..")

from MemeDatasetClipVGG import MemeDatasetClipVGG
from MemeModelCLIPVGG import MemeModelCLIPVGG
from dataset_downloader.load_data import load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import dataset_downloader.load_data as load_data
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


def train_model(model, train_dataloader, val_dataloader, optimizer, losses, vals, num_epochs=10):
    torch.cuda.empty_cache()
    #tqdm(range(num_epochs), position=0, leave=True)
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        train_running_loss = 0.0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
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

            # Zero the parameter gradients
            optimizer.zero_grad()

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

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update running loss
            train_running_loss += loss.item()
        
        # Print the average loss for this epoch
        train_running_loss = train_running_loss / len(train_dataloader)
        losses.append(train_running_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_running_loss:.4f}')
    
        # Validation
        model.eval()
        val_running_loss = 0.0
    
        with torch.no_grad():
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
                
            # Print the average loss for this epoch
            val_running_loss = val_running_loss / len(train_dataloader)
            vals.append(val_running_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Val_Loss: {val_running_loss:.4f}')
            

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
    model.to(device) 

    losses = []
    vals = []
    train_model(model, train_dataloader, val_dataloader, optimizer, losses, vals, num_epochs=15)
    torch.save({
            'epoch': 20,
            'model':model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses[-1],
            }, "model_chekpoints/checkpoint_with_model.pth")