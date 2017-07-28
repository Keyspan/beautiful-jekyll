---
layout: post
title: How does LSA classify documents?
subtitle: Simplest NLP technique 
bigimg: img/ducks.jpg
---

Image you have a bunch of documents that you'd like to classify  based on their topics. The intuitive idea is to pick up keywords of each document, then cluster them into different categories. While you can't pick them up manually, but you could represent each documents using a matrix. The matrix is called term-document matrix or occurrence matrix. Literally, this matrix represents the frequency of each word in each document. More specifically, after you get those documents, e.g. 100 documents, you could build the corresponding vocabulary, like 5000 words. Each document could be represented as a vector with dimension of 5000, with every element representing the frequency of corresponding word. Vice versa, every word could also be represented as a vector with dimension of 100, with element representing the frequency of that word in corresponding document.

Now we have a matrix, looks like:


<!--$$\begin{bmatrix}
1&0&3&5&9&\ldots\\
2&1&5&3&4&\ldots\\
3&6&4&2&1\\
\vdots&\vdots&\vdots&\vdots&\vdots&\ldots\end{bmatrix}$$-->


![](https://ws2.sinaimg.cn/large/006tKfTcgy1fhzh35v8w3j316k0isq5g.jpg)

The LSA assumes that words that are closed in meaning should appear with a similar times among documents. In this matrix, if there is a word 'he', which is the row of the matrix, to be (1,0,3,5,9,...), similary to word 'I'. We're pretty confident that the word 'he' is closed to word 'I'. [wiki](https://en.wikipedia.org/wiki/Latent_semantic_analysis). 

Now how do you classify the documents based on this matrix? We are expecting to reduce the dimension of this matrix, mapping it into lower rank. The singular value decomposition (SVD) 