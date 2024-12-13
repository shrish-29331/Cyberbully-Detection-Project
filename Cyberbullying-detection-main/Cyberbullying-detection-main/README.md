# Cyberbullying-detection
Cyberbullying, poses significant psychological risks, particularly through the use of memes that combine images and text in social media and various otheer platform. This study addresses the complex challenge of detecting cyberbullying in multimodal, code-mixed (Hindi-English) memes. We analyse multitask learning framework that enhances cyberbullying detection by incorporating sentiment, emotion, and sarcasm analysis. A new dataset was made, MultiBully, comprising 5,854 memes annotated with labels for bullying, sentiment, emotion, sarcasm, and harmfulness, was created to support this research, which is analysed bu us. The proposed framework utilizes advanced feature extraction techniques, including RoBERTa for text and VGG19 and CLIP for images, to capture the nuanced interplay of multimodal content. Experimental results demonstrate that our approach significantly improves the accuracy of cyberbullying detection, providing a comprehensive solution to the limitations of traditional text-based methods. This work highlights the importance of multimodal analysis and the integration of affective cues in combating cyberbullying on social media.


# Download Dataset

TO download the dataset run the python file

```
python3 dataset_downloader/download_dataset.py
```

# Run the Base paper Implementation

1. Load the Dataset go to ```dataset_downloader/load_data.py ```  and set the ```img_dir``` and ```excel_dir``` after downloading the dataset
2. run ```basepaper_implementation/train.py``` to train the model
3. run ```basepaper_implementation/validate.py``` to do the validation 


# Run Proposed Approach 1 and 2

1. run ```python3 Proposed_approach_1/main.py```
2. run ```python3 Proposed_approach_2/main.py```

To run different model import them from the `Proposed_approach_1/models/`  or `Proposed_approach_1/models/` and use them accordingly
