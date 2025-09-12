# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

set -e

cd ~

pip install transformers==4.47.1

cd ~/MInference_latest
pip install .

pip install jieba rouge
pip install tree-sitter==0.21.3
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
pip install flash-attn --no-build-isolation


pip install datasets jsonlines fire matplotlib pandas seaborn tqdm
pip install accelerate
pip install jieba mysql-connector-python fuzzywuzzy rouge jsonlines SentencePiece
pip install git+https://github.com/NVIDIA/NeMo.git
pip install nltk  hydra-core wonderwords lightning lhotse  jiwer librosa pyannote-core webdataset editdistance pyannote.metrics tenacity xopen
pip install html2text bs4
python -c 'import nltk; nltk.download("punkt_tab")'
