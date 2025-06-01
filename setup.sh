#!/bin/bash

# Создаем conda environment
conda create -n security_llm python=3.9 -y

# Активируем environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate security_llm

# Устанавливаем базовые пакеты через conda
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c conda-forge transformers datasets accelerate
conda install -y -c conda-forge peft bitsandbytes
conda install -y -c conda-forge wandb tensorboard
conda install -y -c conda-forge scikit-learn pandas numpy
conda install -y -c conda-forge jupyter

# Создаем необходимые директории
mkdir -p data/raw data/processed
mkdir -p models
mkdir -p src/{data,training,evaluation,integration}
mkdir -p notebooks
mkdir -p configs

# Устанавливаем дополнительные пакеты через pip
pip install -r requirements.txt

echo "Окружение успешно настроено!"

conda activate security_llm

echo "Установка завершена!"