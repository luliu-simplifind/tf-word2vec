This is the same model implemented in TensorFlow 2.x. Detailed usage information can be found in the [original README](../README.md).

# Instruction to run word2vec with TensorFlow 2.x

### Installation
1. Require python3 and pip3
2. Install TensorFlow 2.x with pip3
```
pip3 install --upgrade tensorflow
```
3. Check TensorFlow and numpy version
```
pip3 show tensorflow
pip3 show numpy
```

### Build model with imdb review data
1. Download imdb reviews from http://ai.stanford.edu/%7Eamaas/data/sentiment/
2. Unpack the zip file on local. For example, all reviews are available on my loal at: `~/Downloads/aclImdb/train/unsup/` Each file in the folder constains one review. It is most convenient to concatenate a few big files for training.
3. Train the model with complete dataset. Assume now we are in the code dir. The following training specifies the input file as `/tmp/imdb_sample.txt`, where `imdb_sample.txt` is an aggregate of mulitple reviews. The training also designates `/tmp` as output dir.
```
cd tf2.x

python3 run_training.py --filenames=/tmp/imdb_sample.txt --out_dir=/tmp/ --batch_size=64 --window_size=5 --epochs=5
```
4. Examine the training model. In the output folder (i.e., `/tmp`), we should see two files:
* vocab.txt: word vocabulary, one word per line
* syn0_final.npy: word embeddings, numpy array of shape
5. Test training results
```
python3 demo_word_similarity.py /tmp funny
```
In above command, `/tmp` is the output folder from training step. And `funny` is the query to find most similar words.