import audeer
import audonnx
import numpy as np

import os
import json
import librosa
from tqdm import tqdm
import pdb
import argparse

cache_root = audeer.mkdir('cache')
model_root = audeer.mkdir('model')

model = audonnx.load(model_root)

sampling_rate = 16000

import numpy as np
import librosa

# 加载音频文件

logits = []

data_path = '/home/hml522/mydata/LJSpeech-1.1/wavs'
os.makedirs('emotion_demension_finer_ljspeech', exist_ok=True)
os.makedirs('emotion_demension_coarse_ljspeech', exist_ok=True)

def main():
    # logits_by_speaker_emotion[speaker] = {}
    for file in tqdm(os.listdir(data_path)):
        
        if file[-4:] != '.wav':
            continue
        
        basename = file[:-4]

        signal, sr = librosa.load(os.path.join(data_path, file), sr=sampling_rate)

        # 设置窗口大小和步长
        window_size = int(0.25 * sampling_rate)  # 0.25 秒的窗口大小
        hop_size = int(0.125 * sampling_rate)    # 0.125 秒的步长

        # 提取滑动窗口
        windows = []
        finer_logits = []

        for start in range(0, len(signal) - window_size + 1, hop_size):
            end = start + window_size
            window = signal[start:end]
            windows.append(window)
            logit = model(window, sampling_rate)['logits']
            finer_logits.append(logit)

        # 将结果转换为 NumPy 数组
        windows = np.array(windows)
        # 音素级别
        finer_logits = np.array(finer_logits) - 0.5
        finer_logits = np.squeeze(finer_logits)
        # 粗粒度
        coarse_logits = model(signal, sampling_rate)['logits'] - 0.5

        logits.append(coarse_logits)

        finer_save_path = os.path.join('emotion_demension_finer_ljspeech')
        os.makedirs(finer_save_path, exist_ok=True)
        coarse_save_path = os.path.join('emotion_demension_coarse_ljspeech')
        os.makedirs(coarse_save_path, exist_ok=True)

        if not os.path.exists(os.path.join(finer_save_path, f'{basename}.npy')):
            np.save(os.path.join(finer_save_path, f'{basename}.npy'), coarse_logits)
        if not os.path.exists(os.path.join(coarse_save_path, f'{basename}.npy')):
            np.save(os.path.join(coarse_save_path, f'{basename}.npy'), finer_logits)

    mean_logits = np.mean(logits, axis=0)
        
    with open('mean_logits_ljspeech.json', 'w') as f:
        json.dump(mean_logits, f, indent=4)
    print()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--speaker", type=str)
    # args = parser.parse_args()
    # main(args)
    main()

