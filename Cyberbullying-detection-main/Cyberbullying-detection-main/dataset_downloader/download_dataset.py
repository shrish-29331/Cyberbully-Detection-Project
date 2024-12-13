import subprocess
import os
from tqdm import tqdm

def download_file(url, output_path):
    try:
        result = subprocess.run(['wget','--no-check-certificate', url, '-O', output_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr.decode()}")

def run_command(command):
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr.decode()}")


def main():
    download_file('https://iitgoffice-my.sharepoint.com/:u:/g/personal/sjana_iitg_ac_in/EWdMrS9zHgBHnz4TTckw14kB14O8j0IbR_D-fBowyw7T7A?e=8H4Wpw&download=1', 'bully_data.zip')
    
    # Remove the directory if it exists
    if os.path.exists('bully_data'):
        run_command('rm -rf bully_data')

    # Create directories
    os.makedirs('bully_data/data')

    # Unzip files
    with tqdm(total=2) as pbar:
        run_command('unzip bully_data.zip -d bully_data/data/')
        pbar.update(1)
        #run_command('unzip Cyberbully_corrected_emotion_sentiment_v2.zip -d bully_data/')
        #pbar.update(1)

if __name__ == '__main__':
    main()