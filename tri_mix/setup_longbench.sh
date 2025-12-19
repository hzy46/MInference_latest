# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

set -e

cd ~

pip install transformers==4.47.1


cd ~/MInference_latest
pip install -e . --no-build-isolation

pip install jieba rouge
pip install tree-sitter==0.21.3
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
pip install flash-attn --no-build-isolation

pip install datasets jsonlines fire matplotlib pandas seaborn tqdm SentencePiece
pip install jieba mysql-connector-python fuzzywuzzy rouge
pip install datasets==3.6.0
