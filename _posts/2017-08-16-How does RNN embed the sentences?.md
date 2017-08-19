---
layout: post
title: How does RNN embed the sentences?
subtitle: RNN approach to understand sentences
bigimg: /img/ducks.jpg
---

> We have already figured out ways of word2vec, LDA, and LSA to analyze words, sentences, documents. But, obviously ,those are not enough for us the understand specific sentence. We'd like to embed a sentence and then we'll be able to do a bunch of exciting stuffs.

This post, we will talk about RNN approach, speicifically in TensorFlow. I strongly recommend you to carefully read and imitate the codes, which are super helpful to understand RNN. [WildML](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)

First, old rules, theory.

#### Recurrent Neural Network

The word in the first place will influence the chance of what the second word will be. In other words, there are dependency between words. In Recurrent Neural Network, we're aiming to transfer the relations from the start to the end for specific sentence. Just look at the graph below, <img src="https://latex.codecogs.com/svg.latex?x_t,&space;x_{t-1}" align = 'center' /> represent words in a sentence, e.g. "I", "will", whose relationship could be represented by multiplying matrice and applying transforming functions. The relationship are transferred from the first word to the last word by considering the hidden state <img src="https://latex.codecogs.com/svg.latex?h_{t}" align = 'center'/> of current word to the next word's hidden state <img src="https://latex.codecogs.com/svg.latex?h_{t+1}" align = 'center'/>. The output <img src="https://latex.codecogs.com/svg.latex?o_t" align = 'center'/> is <img src="https://latex.codecogs.com/svg.latex?vocab\_size" /> vector, used for predicting next word from <img src="https://latex.codecogs.com/svg.latex?vocab\_size" />  target words.

![](https://ws1.sinaimg.cn/large/006tKfTcgy1fimam2z7quj30ok08u0tb.jpg)


The detailed mathematical formula are as follows:

**Still working on it...**