set -e

cd ~

pip install transformers


cd ~/MInference_latest
pip install .

pip install jieba rouge
pip install tree-sitter==0.21.3
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
pip install flash-attn --no-build-isolation

pip install datasets jsonlines fire matplotlib pandas seaborn tqdm SentencePiece
pip install jieba mysql-connector-python fuzzywuzzy rouge