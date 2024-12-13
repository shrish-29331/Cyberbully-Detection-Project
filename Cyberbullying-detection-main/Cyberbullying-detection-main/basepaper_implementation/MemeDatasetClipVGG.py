import torch
import os
from torchvision import transforms, models
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset
from PIL import Image
import pretrained_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#clip_model, vgg19, roberta_model, tokenizer, transform, device = pretrained_models.get_models()

class MemeDatasetClipVGG(Dataset):
    def __init__(self, dataframe, transform, clip_model, vgg19):
        self.dataframe = dataframe
        self.transform = transform
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
        self.clip_model = clip_model
        self.vgg19 = vgg19
        #self.resnet = resnet50
        
        self.img_folder = '/kaggle/working/bully_data/data/bully_data'
        
        # Define label mappings
        self.text_label_mapping = {
            "Bully": 1,
            "Nonbully": 0
        }
        
        self.sentiment_mapping = {
            "Positive":1,
            "Neutral": 0,
            "Negative": 2
        }
        
        self.emotion_mapping = {
            "Disgust": 0,
            "Ridicule": 1,
            "Sadness": 2,
            "Surprise": 3,
            "Anticipation": 4,
            "Angry": 5,
            "Happiness": 6,
            "Other": 7,
            "Trust": 8,
            "Fear": 9
        }
        
        self.sarcasm_mapping = {
            "Yes": 1,
            "No": 0
        }
        
        self.harmful_score_mapping = {
            "Harmless": 0,
            "Partially-Harmful": 1,
            "Very-Harmful": 2
        }
        
        self.target_mapping = {
            "Individual": 0,
            "Society": 1,
            "Organization": 2,
            "Community": 3
        }
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        #print(f"<*>Loading {idx} th Data point-----")
        # Load image
        img_name = self.dataframe.iloc[idx]['Img_Name']
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        img_tensor = torch.tensor(image.unsqueeze(0)).to(device)
        #img_tensor = self.transform(image).unsqueeze(0).to(device)  # Transform and add batch dimension
    
        
        # 2. Extract image features using CLIP and VGG19
        with torch.no_grad():
            #print("Loading CLIP model")
            image_clip_input = self.clip_model.encode_image(img_tensor).float().to(device)  #(batch_size, 512)
            #print("Loading VGG19 model")
            image_vgg_feature = self.vgg19(img_tensor).view(-1).float().to(device)  # Flattened VGG19 features (batch_size, 1024)
            #print("|------VGG Features Done")
            
        # 3. Process text with CLIP and RoBERTa
        text = self.dataframe.iloc[idx]['Img_Text']
        text = text.replace("\n", "")
        #print(f" Datapoint {idx}: Img_Text: {text}  ")
        #print("|------IMG_Text Loaded Success")
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=77)
        #print("|------Tokenizer Success")
        input_ids = tokens['input_ids'].to(device)
        #print("|------Input IDs Success")
        attention_mask = tokens['attention_mask'].to(device)
        #print("|------Attention Mask Success")
        
        #clip_tokens = clip.tokenize([text]).to(device)
        

        with torch.no_grad():
            #print("Loading CLIP text Encoding......")
            #print(input_ids)
            text_clip_input = self.clip_model.encode_text(input_ids).float().to(device)  #(batch_size, 512)
            #print("    |------CLIP Text input Encoding Success")
            text_roberta_output = self.roberta_model(input_ids, attention_mask=attention_mask)
            #print("    |------RoBERTa output with attention Mask Success")
            text_roberta_embedding = text_roberta_output.last_hidden_state[:, 0, :].float().to(device)  #(batch_size, 768)
            #print("|------Roberta embedding Success")
        
        # Load and tokenize text
        #text = self.dataframe.iloc[idx]['Img_Text']
        #inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        
        # Get labels and apply mappings
        sentiment_label = torch.tensor(self.sentiment_mapping[self.dataframe.iloc[idx]['Sentiment']], dtype=torch.long)
        emotion_label = torch.tensor(self.emotion_mapping[self.dataframe.iloc[idx]['Emotion']], dtype=torch.long)
        sarcasm_label = torch.tensor(self.sarcasm_mapping[self.dataframe.iloc[idx]['Sarcasm']], dtype=torch.float)  # Binary sarcasm
        bully_label = torch.tensor(self.text_label_mapping[self.dataframe.iloc[idx]['Img_Label']], dtype=torch.long)  # Bully detection
        harmful_score_label = torch.tensor(self.harmful_score_mapping[self.dataframe.iloc[idx]['Harmful_Score']], dtype=torch.long)
        target_label = torch.tensor(self.target_mapping[self.dataframe.iloc[idx]['Target']], dtype=torch.long)
        #print("|------Processed------|")
        # 5. Prepare the sample dictionary
        sample = {
            "id": idx,
            "image": image, # 3x224x224
            "image_clip_input": image_clip_input,  # CLIP image features (1x512-dim)
            "image_vgg_feature": image_vgg_feature,  # VGG19 image features (flattened 1x25088-dim)
            "text_clip_input": text_clip_input,  # CLIP text features (1x512-dim)
            "text_roberta_embedding": text_roberta_embedding,  # RoBERTa text embedding (1x768-dim)
            "bully_label": bully_label,  # Cyberbullying label
            "sentiment": sentiment_label,  # Sentiment label
            "sarcasm": sarcasm_label,  # Sarcasm label
            "emotion": emotion_label,  # Emotion label
            "harmful-score": harmful_score_label,  # Harmful score
            "target": target_label  # Target variable
        }
        
        #print("|------Sampled------|")
        
        return sample