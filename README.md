# Learning Cross-Lingual Phonological and Orthagraphic Adaptations [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

This repo hosts the code necessary to reproduce the results of our paper, formerly titled as [**Neural machine translation based
word transduction mechanisms for Low Resource Languages**](https://arxiv.org/abs/1811.08816) (recently accepted at the Journal of Language Modelling). 

![enc-dec](https://github.com/Saurav0074/nmt-based-word-transduction/blob/master/dataset_cognates/enc_dec.png)
***

## Generating char2vec from pre-trained Hindi fastText embeddings

The pre-trained Hindi character vectors can be downloaded from [here](https://github.com/Saurav0074/nmt-based-word-transduction/tree/master/dataset_char2vec). This repo contains two such methods for generating these character embeddings:

1. Running the file [`generate_char2vec.py`](https://github.com/Saurav0074/nmt-based-word-transduction/blob/master/preprocess/generate_char2vec.py) generates the character vectors for **71 Devanagari characters** from the pre-trained word vectors. The outputs can be found in `char2vec.txt`.
2. Running the file [`char_rnn.py`](https://github.com/Saurav0074/nmt-based-word-transduction/blob/master/character-model/char_rnn.py) trains a language model over the `hindi-wikipedia-articles-55000` (i.e., generating the 30th character given the sequence of 29 consecutive characters). The embedding weights are then retained to extract the character-level embeddings.

## Models Used

We experimented with four variants of sequence-to-sequence models for our project:
- **Peeky Seq2seq Model**: Run the file `peeky_Seq2seq.py`. The implementation is based on [Sequence to Sequence Learning with Keras](https://github.com/farizrahman4u/seq2seq).

- **Alignment Model (AM)**: Run the file `attentionDecoder.py`. Following the work of _Bahdanau et al._ [1], the file `attention_decoder.py` contains the custom Keras layer based on Tensorflow backend. The original implementation can be found [here](https://github.com/datalogue/keras-attention/blob/master/models/custom_recurrents.py). A good blog post guiding the use of this implementation can be found [here](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/).
- **Heirarchical Attention Model (HAM)**: Run the file `attentionEncoder.py`. Inspired from the work of _Yang et al._ [2] Original implementation can be found [here](https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2).

- **Transformer Network**: `generate_data_for_tensor2tensor.py` generates the data as required by the Transformer network. [3] The data is required while registering your own database (See [this](https://tensorflow.github.io/tensor2tensor/new_problem.html) for further reading). For a detailed look at installation and usage, visit their official github [page](https://github.com/tensorflow/tensor2tensor).

## Evaluation metrics
- `bleu_score.py` measures the BLEU score between the transduced and the actual Bhojpuri words averaged over the entire output file.

- `word_accuracy.py` simply measures the proportion of correctly transduced words in the output file.

- `measure_distance.py` measures the Soundex score similarity between the actual and transduced Bhojpuri word pairs, averaged over the output file. A good blog post explaining the implementation can be found [here](http://thottingal.in/blog/2009/07/26/indicsoundex/).

## Citation 

If our code was helpful in your research, consider citing our work:

```
@article{jha2018neural,
  title={Neural Machine Translation based Word Transduction Mechanisms for Low-Resource Languages},
  author={Jha, Saurav and Sudhakar, Akhilesh and Singh, Anil Kumar},
  journal={arXiv preprint arXiv:1811.08816},
  year={2018}
}
```

***
# References 
[1] Bahdanau, D., Bengio, Y., & Cho, K. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. CoRR, abs/1409.0473.

[2] Dyer, C., He, X., Hovy, E.H., Smola, A.J., Yang, Z., & Yang, D. (2016). Hierarchical Attention Networks for Document Classification. HLT-NAACL.

[3] Gomez, A.N., Jones, L., Kaiser, L., Parmar, N., Polosukhin, I., Shazeer, N., Uszkoreit, J., & Vaswani, A. (2017). Attention is All you Need. NIPS.
