# MISRNet: Lightweight Neural Vocoder Using Multi-Input Single Shared Residual Blocks

This repository provides **unofficial** pytorch implementation of HiFi-GAN with fast MISR.

[paper(Interspeech 2022 archive)](https://www.isca-speech.org/archive/interspeech_2022/kaneko22_interspeech.html)  
[**Official** demo page](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/misrnet/index.html)

**Abstract :**
Neural vocoders have recently become popular in text-tospeech synthesis and voice conversion, increasing the demand
for efficient neural vocoders. One successful approach is HiFiGAN, which archives high-fidelity audio synthesis using a relatively small model. This characteristic is obtained using a
generator incorporating multi-receptive field fusion (MRF) with
multiple branches of residual blocks, allowing the expansion of
the description capacity with few-channel convolutions. However, MRF requires the model size to increase with the number
of branches. Alternatively, we propose a network called MISRNet, which incorporates a novel module called multi-input single shared residual block (MISR). MISR enlarges the description capacity by enriching the input variation using lightweight
convolutions with a kernel size of 1 and, alternatively, reduces
the variation of residual blocks from multiple to single. Because
the model size of the input convolutions is significantly smaller
than that of the residual blocks, MISR reduces the model size
compared with that of MRF. Furthermore, we introduce an implementation technique for MISR, where we accelerate the processing speed by adopting tensor reshaping. We experimentally applied our ideas to lightweight variants of HiFi-GAN and
iSTFTNet, making the models more lightweight with comparable speech quality and without compromising speed.

## Pre-requisites
1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt)
4. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).
And move all wav files to `LJSpeech-1.1/wavs`


## Training
```
python train.py --config config_v1.json
```
To train V2 Generator, replace `config_v1.json` with `config_v2.json`.<br>
Checkpoints and copy of the configuration file are saved in `cp_misrnet` directory by default.<br>
You can change the path by adding `--checkpoint_path` option.

## Fine-Tuning
1. Generate mel-spectrograms in numpy format using [Tacotron2](https://github.com/NVIDIA/tacotron2) with teacher-forcing.<br/>
The file name of the generated mel-spectrogram should match the audio file and the extension should be `.npy`.<br/>
Example:
    ```
    Audio File : LJ001-0001.wav
    Mel-Spectrogram File : LJ001-0001.npy
    ```
2. Create `ft_dataset` folder and copy the generated mel-spectrogram files into it.<br/>
3. Run the following command.
    ```
    python train.py --fine_tuning True --config config_v1.json
    ```
    For other command line options, please refer to the training section.


## Inference from wav file
1. Make `test_files` directory and copy wav files into the directory.
2. Run the following command.
    ```
    python inference.py --checkpoint_file [generator checkpoint file path]
    ```
Generated wav files are saved in `generated_files` by default.<br>
You can change the path by adding `--output_dir` option.


## Inference for end-to-end speech synthesis
1. Make `test_mel_files` directory and copy generated mel-spectrogram files into the directory.<br>
You can generate mel-spectrograms using [Tacotron2](https://github.com/NVIDIA/tacotron2), 
[Glow-TTS](https://github.com/jaywalnut310/glow-tts) and so forth.
2. Run the following command.
    ```
    python inference_e2e.py --checkpoint_file [generator checkpoint file path]
    ```
Generated wav files are saved in `generated_files_from_mel` by default.<br>
You can change the path by adding `--output_dir` option.


## Acknowledgements
We referred to [HiFi-GAN](https://github.com/jik876/hifi-gan) to implement this.

