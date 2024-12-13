class Image_text(nn.Module):
    def __init__(self):
        super(Image_text, self).__init__()

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
        self.sentiment_head = nn.Linear(512, 3)  # Sentiment: 3 classes
        self.emotion_head = nn.Linear(512, 10)    # Emotion: 6 classes
        self.sarcasm_head = nn.Linear(512, 1)    # Sarcasm: binary classification
        self.bully_head = nn.Linear(512, 2)      # Cyberbullying: 2 classes
        self.harmful_head = nn.Linear(512, 3)
        self.target_head = nn.Linear(512, 4)
    def forward(self, image, text_input_ids, text_attention_mask):
        # Visual features
        img_features = self.resnet(image)

        # Textual features
        text_outputs = self.roberta(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_features = text_outputs.pooler_output

        # Concatenate the visual and textual features
        combined_features = torch.cat((img_features, text_features), dim=1)

        # Shared layers
        shared_out = self.fc_shared(combined_features)

        # Task-specific outputs
        sentiment_out = self.sentiment_head(shared_out)
        emotion_out = self.emotion_head(shared_out)
        sarcasm_out = torch.sigmoid(self.sarcasm_head(shared_out))
        bully_out = self.bully_head(shared_out)
        harmful_out = self.harmful_head(shared_out)
        target_out = self.target_head(shared_out)

        return sentiment_out, emotion_out, sarcasm_out, bully_out, harmful_out, target_out