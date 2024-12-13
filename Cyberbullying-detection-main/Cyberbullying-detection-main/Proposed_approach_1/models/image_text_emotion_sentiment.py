class Image_text_emotion_sentiment(nn.Module):
    def __init__(self):
        super(Image_text_emotion_sentiment, self).__init__()

        # Visual branch (CNN)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final classification layer

        # Textual branch (RoBERTa)
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        # Shared fully connected layers
        self.fc_shared = nn.Sequential(
            nn.Linear(2048 + 768, 1024),  # Concatenation of visual and textual features
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),  # Deeper layer
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Task-specific heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # Sentiment: 3 classes
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)  # Emotion: 3 classes
        )

        self.sarcasm_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),  # Sarcasm: binary classification
            nn.Sigmoid()  # Ensures output is in (0, 1)
        )
        # self.bully_head = nn.Linear(512, 2)  # Bully: binary classification
        self.bully_fc = nn.Sequential(
            nn.Linear(10 + 3 + 512, 256),  # Input: all task outputs + shared features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 2 classes for bully
        )
        # self.harmful_head = nn.Linear(512, 3)  # Harmful score: 3 classes
        self.harmful_fc = nn.Sequential(
            nn.Linear(10 + 3+ 512, 256),  # Input: all task outputs + shared features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 3 classes for harmful
        )
        # Final target head
        self.target_fc = nn.Sequential(
            nn.Linear(10 + 3 +  512, 256),  # Input: all task outputs + shared features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)  # 4 classes for Target
        )

    def forward(self, image, text_input_ids, text_attention_mask):
        # Visual features
        img_features = self.resnet(image)

        # Textual features
        text_outputs = self.roberta(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_features = text_outputs.pooler_output

        # Concatenate the visual and textual features
        combined_features = torch.cat((img_features, text_features), dim=1)

        # Shared features
        shared_out = self.fc_shared(combined_features)

        # Task-specific predictions
        sentiment_out = self.sentiment_head(shared_out)
        emotion_out = self.emotion_head(shared_out)
        sarcasm_out = torch.sigmoid(self.sarcasm_head(shared_out))  # Binary
        # bully_out = self.bully_head(shared_out)
        # harmful_out = self.harmful_head(shared_out)

        # Concatenate all task outputs with shared features for target prediction
        aux_features = torch.cat((
            emotion_out,
            sentiment_out,
            shared_out  # Shared features
        ), dim=1)

        # Final target prediction
        bully_out = self.bully_fc(aux_features)
        harmful_out = self.harmful_fc(aux_features)
        target_out = self.target_fc(aux_features)

        return sentiment_out, emotion_out, sarcasm_out, bully_out, harmful_out, target_out
