import torch
import torch.nn as nn
import torch.nn.functional as F

from ImageClipModel import ImageClipModel
from ImageVGGDenseModel import ImageVGGDenseModel
from SelfAttentionBiLSTM import SelfAttentionBiLSTM
from TextClipModel import TextClipModel
from TextRobertaModel import TextRobertaModel

class MemeModelCLIPVGG(nn.Module):
    def __init__(self,img_clip_inp_dim, vgg_inp_dim, text_clip_inp_dim, text_roberta_inp_dim):
        super(MemeModelCLIPVGG,self).__init__()
        
        self.image_clip_model = ImageClipModel(img_clip_inp_dim)
        self.vgg_dense =  ImageVGGDenseModel(vgg_inp_dim)
        self.text_clip_model = TextClipModel(text_clip_inp_dim)
        self.text_roberta_model = TextRobertaModel(text_roberta_inp_dim)
        #self.resnet_model = ResNetPreModel()
        
        self.attention_bilstm_model = SelfAttentionBiLSTM(text_roberta_inp_dim)
        
        # Load pre-trained RoBERTa model
        #self.text_roberta_model = RobertaModel.from_pretrained('roberta-base') # ->768
        
        # Shared fully connected layers
        self.fc_shared = nn.Sequential(
            nn.Linear(128 + 128 + 128 + 512 + 512, 1024), # one 512 removed as resnet is not used
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Define fully connected layers for each task
        self.fc_sentiment = nn.Linear(512, 3)  # 3 classes for sentiment
        self.fc_emotion = nn.Linear(512, 10)  # 10 classes for emotion
        self.fc_sarcasm = nn.Linear(512, 1)  # Binary classification for sarcasm
        self.fc_bully = nn.Linear(512, 2)  # Binary classification for bully detection
        self.fc_harmful_score = nn.Linear(512, 3)  # 3 classes for harmful score
        self.fc_target = nn.Linear(512, 4)  # 4 classes for target
        
    
    def forward(self, image, image_clip_input, image_vgg_feature, text_clip_input, text_roberta_embedding):
    
        #use BERT + VGG19 combination
        
        img_clip_out = self.image_clip_model(image_clip_input) # 512 ->128
        text_clip_out = self.text_clip_model(text_clip_input) #512 ->128
        text_roberta_out = self.text_roberta_model(text_roberta_embedding) # 768 -> 128
        #img_resnet_out = self.resnet_model(image) # 3x224x224 -> 512
        vgg19_out = self.vgg_dense(image_vgg_feature) #25088 -> 512 # in_CI
        
        self_atten_bilstm_out = self.attention_bilstm_model(text_roberta_embedding) #768 -> 512
        #128 + 128+ 128+ 512 + 512 +512
        
#         print("img_clip_out shape:", img_clip_out.shape)
#         print("text_clip_out shape:", text_clip_out.shape)
#         print("text_roberta_out shape:", text_roberta_out.shape)
#         print("img_resnet_out shape:", img_resnet_out.shape)
#         print("vgg19_out shape:", vgg19_out.shape)
#         print("self_atten_bilstm_out shape:", self_atten_bilstm_out.shape)
        
        # Add a batch dimension to vgg19_out
        # if vgg19_out.dim() == 1:  # Check if it has only one dimension
            # vgg19_out = vgg19_out.unsqueeze(0)  # Shape: (1, 512)
            
        # Squeeze the second dimension to make all tensors 2D
        img_clip_out = img_clip_out.squeeze(1)  # Shape: (8, 128)
        text_clip_out = text_clip_out.squeeze(1)  # Shape: (8, 128)
        text_roberta_out = text_roberta_out.squeeze(1)  # Shape: (8, 128)
        
        combined_features = torch.cat((img_clip_out, text_clip_out, text_roberta_out, vgg19_out, self_atten_bilstm_out), dim = 1)
        
        features = self.fc_shared(combined_features)
        
        # Pass through fully connected layers for each task
        sentiment_output = self.fc_sentiment(features)
        emotion_output = self.fc_emotion(features)
        sarcasm_output = self.fc_sarcasm(features)
        bully_output = self.fc_bully(features)
        harmful_score_output = self.fc_harmful_score(features)
        target_output = self.fc_target(features)
        
        # Task-specific outputs with softmax or sigmoid
#         sentiment_output = F.softmax(self.fc_sentiment(features), dim=1)  # 3-class softmax
#         emotion_output = F.softmax(self.fc_emotion(features), dim=1)      # 10-class softmax
#         sarcasm_output = torch.sigmoid(self.fc_sarcasm(features))         # Binary sigmoid
#         bully_output = F.softmax(self.fc_bully(features), dim=1)          # Binary softmax
#         harmful_score_output = F.softmax(self.fc_harmful_score(features), dim=1)  # 3-class softmax
#         target_output = F.softmax(self.fc_target(features), dim=1)        # 4-class softmax
        
        return sentiment_output, emotion_output, sarcasm_output, bully_output, harmful_score_output, target_output