import torch
import subprocess
import os
from tqdm import tqdm
import numpy as np
from torchvision import transforms, models
from transformers import RobertaTokenizer, RobertaModel


MODELS = {
     "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
     "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
     "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
     "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
 }

def download_file(url, output_path):
    try:
        result = subprocess.run(['wget', url, '-O', output_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr.decode()}")

def run_command(command):
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr.decode()}")

def download_models():
    download_file(MODELS["ViT-B/32"], 'clip_model.pt')

def get_models():
    
    download_models()

    clip_model = torch.jit.load("clip_model.pt").cuda().eval()
    input_resolution = clip_model.input_resolution.item()
    context_length = clip_model.context_length.item()
    vocab_size = clip_model.vocab_size.item()

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    # Load CLIP, VGG19, and Roberta models
    clip_model = torch.jit.load("clip_model.pt").cuda().eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VGG19 Model
    vgg19 = models.vgg19(pretrained=True).features.to(device).eval()
    resnet18 = models.resnet50(weights="IMAGENET1K_V1").to(device).eval()

    # Load RoBERTa TOkenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)

    # Image transformation for VGG19 and CLIP
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return clip_model, vgg19, roberta_model, tokenizer, transform, device
