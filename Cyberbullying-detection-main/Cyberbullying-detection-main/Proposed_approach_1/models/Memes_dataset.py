class MemeDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.img_folder = '/kaggle/input/multibully/bully_data'
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

        # self.emotion_mapping = {
        #     "Positive":1,
        #     "Neutral": 0,
        #     "Negative": 2
        # }
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
        # Load image
        img_name = self.dataframe.iloc[idx]['Img_Name']
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Load and tokenize text
        text = self.dataframe.iloc[idx]['Img_Text']
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)

        # Get labels and apply mappings
        sentiment_label = torch.tensor(self.sentiment_mapping[self.dataframe.iloc[idx]['Sentiment']], dtype=torch.long)
        emotion_label = torch.tensor(self.emotion_mapping[self.dataframe.iloc[idx]['Emotion']], dtype=torch.long)
        sarcasm_label = torch.tensor(self.sarcasm_mapping[self.dataframe.iloc[idx]['Sarcasm']], dtype=torch.float)  # Binary sarcasm
        bully_label = torch.tensor(self.text_label_mapping[self.dataframe.iloc[idx]['Img_Label']], dtype=torch.long)  # Bully detection
        harmful_score_label = torch.tensor(self.harmful_score_mapping[self.dataframe.iloc[idx]['Harmful_Score']], dtype=torch.long)
        target_label = torch.tensor(self.target_mapping[self.dataframe.iloc[idx]['Target']], dtype=torch.long)

        return image, inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), sentiment_label, emotion_label, sarcasm_label, bully_label, harmful_score_label, target_label
