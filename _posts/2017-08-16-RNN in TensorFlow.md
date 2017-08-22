---
layout: post
title: RNN in TensorFlow
subtitle: RNN approach to understand sentences
bigimg: /img/ducks.jpg
---

> We have already figured out ways of word2vec, LDA, and LSA to analyze words, sentences, documents. But, obviously ,those are not enough for us the understand specific sentence. We'd like to embed a sentence and then we'll be able to do a bunch of exciting stuffs.


#### Recurrent Neural Network

The word in the first place will influence the chance of what the second word will be. In other words, there are dependency between words. In Recurrent Neural Network, we're aiming to transfer the relations from the first word to the last word. Just look at the graph below, <img src="https://latex.codecogs.com/svg.latex?x_t,&space;x_{t-1}, x_{t+1}\in R^{vocab\_size}" align = 'center' /> are embedding of words in a sentence, e.g. *"I", "will", "be"*, whose relationship could be built through matrice <img src="https://latex.codecogs.com/svg.latex?\mathbf{U, V, W}" align = 'center' /> and transforming(activation) functions. The relationship between the *"I"* to the *"will"* is built through multiplying hidden state <img src="https://latex.codecogs.com/svg.latex?h_{t-1}" align = 'center'/> with matrix <img src="https://latex.codecogs.com/svg.latex?\mathbf{V}" align = 'center' /> and passing to hidden state <img src="https://latex.codecogs.com/svg.latex?h_{t}" align = 'center'/>. By multipying the matrix <img src="https://latex.codecogs.com/svg.latex?\mathbf{W}" align = 'center' />, we get the output <img src="https://latex.codecogs.com/svg.latex?o_t\in R^{vocab\_size}" align = 'center'/> vector, used for predicting next word (right word should be "be") from <img src="https://latex.codecogs.com/svg.latex?vocab\_size", align ='center /> target words. It's like classfication problem with  <img src="https://latex.codecogs.com/svg.latex?vocab\_size", align ='center'/> classes.

![](https://ws2.sinaimg.cn/large/006tKfTcgy1fis14tdkjsj30pa082js3.jpg)


The detailed and clear mathematical formula are as follows:

<img src="https://latex.codecogs.com/svg.latex? h_t&space;=&space;tanh(\mathbf{U}x_t&space;&plus;&space;\mathbf{V}h_{t-1})" align = 'center'/>

<img src="https://latex.codecogs.com/svg.latex?  o_t&space;=&space;softmax(\mathbf{W}h_t)" align = 'center'/>

The activation function is default <img src="https://latex.codecogs.com/svg.latex?tanh(.)" align = "center"/>, you can also apply other functions. The hidden states are passed from the first word <img src="https://latex.codecogs.com/svg.latex?h(1)" align = 'center'/> to the last word <img src="https://latex.codecogs.com/svg.latex?h(T)" align = 'center'/>.

#### RNN script

First, let's go through the R script to better understand how RNN works in codes. The strongly recommended website is [WildML](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/), which is super helpful to understand RNN. I learnt from this website by imitating the codes.

In that post, there are several steps for RNN script:

* Clean the data
	* Tokenize the sentences
	* Remove low frequency words
	* Add unknown symbol, start symbol and end symbol to sentences
	* Embed the sentences 
* Build the model
	* Forward propagation
	* Calculate the loss
	* Train model with SGD and Backpropagation through time(BPTT)

##### Clean the data

First, we talk about the first part -cleaning the data.
For tokenizing, we can use the  `nltk.tokenize.ToktokTokenizer()` method in `nltk` class. The codes are like:

```
from nltk.tokenize import ToktokTokenizer

tok = ToktokTokenizer()

tokenized_sens = [tok.tokenize(sen) for sen in sentences] 
        
```

There are some words that only occur like two, or three times, which are not useful for prediction. We can remove unfrequent words and then train rnn models and predict. The codes are like:

```
frequency_words = nltk.FreqDist(itertools.chain(*tokenized_sens))
        
size = len(list(set(itertools.chain(*(tokenized_sens)))))
        
vocabs = start_vocabs + [w[0] for w in frequency_words.most_common(int(size*ratio))]
```
The `nltk.FreqDist` function could produce pairs of words and corresponding frequency. In the code, I use the ratio to describe how much words I'd like to keep from whole words. Then the corresponding method `most_common()` could help us keep the most frequent words. Also, since I will add some symbols to the sentences when I embed the them later, I will add these symbols to vocabulary here, e.g. "_unk" representing unknown words after removing the unfrequent words.

Now, we use the one-hot vector to initially embed the sentences. Also, like in word2vec, we only use the index of words in vocabulary, since multiplying the matrix with one-hot vector is same as extracting specific column of the matrix. So why we add these 4 symbols? The "start" and "end" symbol is helping us to start predict a sentence and stop the prediction when we test the model. The "unknown" symbol is representing all unfrequency words we delete. The "pad" symbol meas there is no word in this position and don't calculate the loss or predict for this word. We have to pad the sentences to keep the sentences in the same batch with same length, due to the matrix calculation.

```
word_to_index = dict([(w,i) for i, w in enumerate(vocabs)])

# add symbols to sentences
tokenized_sens = [['_start'] + [w if w in vocabs else '_unk' for w in sen]  + ['_end']  for sen in tokenized_sens]

# embed trainning data
x_train = [[word_to_index[w] for w in sen[:-1]] for sen in tokenized_sens]

```
##### Build the model

Forward propagation is easy, where we just pass the parameters into the model and calculate the outputs.
The codes (learn from [wildml-forward-propagation](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)) are like:

```
def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    h = np.zeros((T + 1, self.hidden_dim))
    h[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros((T, self.word_dim))
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        h[t] = np.tanh(self.U[:,x[t]] + self.W.dot(h[t-1]))
        o[t] = softmax(self.V.dot(h[t]))
    return [o, h]
``` 

Since we have the outputs, we can define the prediction function (still [wildml-predict](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)).

```
def predict(self, x):
    # Perform forward propagation
    # and return index of the highest score    
    o, h = self.forward_propagation(x)
    return np.argmax(o, axis=1)
```

Now if we want the train the model, we are aiming to decrease the loss. Here, we consider the  cross-entropy loss, which is the negative maximum log-likelihood. We can define the loss function [wildml-loss](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/) like below:

```
def calculate_total_loss(self, x, y):
    L = 0
    # For each sentence...
    for i in np.arange(len(y)):
        o, h = self.forward_propagation(x[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L
 
def calculate_loss(self, x, y):
    # Divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))
    return self.calculate_total_loss(x,y)/N
```

If we'd like train the model, we have to known the gradient of loss function so that we can decrease the loss. However, there are parameters that dependent on each other throuh time. Remember, we pass the hidden state <img src="https://latex.codecogs.com/svg.latex?h_{t-1}" align = 'center' /> to <img src="https://latex.codecogs.com/svg.latex?h_{t}" align = 'center' />. The <img src="https://latex.codecogs.com/svg.latex?\mathbf{U}, \mathbf{V}, \mathbf{W}" align = 'center' /> exisit in both <img src="https://latex.codecogs.com/svg.latex?h_{t-1}" align = 'center' /> and <img src="https://latex.codecogs.com/svg.latex?h_{t}" align = 'center' />. That's what BPTT handles. BPTT is a step further of Backpropagation. The strongly recommended paper of backpropagation is here, [Backpropagarion](https://www.cs.swarthmore.edu/~meeden/cs81/s10/BackPropDeriv.pdf) which is a derivation of backpropagation. Ok, maybe you don't like that, but it helps me to figure out what's happening. The illustration of BPTT is here [BPTT](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/), and the derivation is attached here, [BPTT derivation](https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf) LOL...
The codes below are also from the [WildML-BPTT](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/). I am just trying to read the codes with you. 

```
def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, h = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], h[t].T)
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (h[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, h[bptt_step-1])              
            dLdU[:,x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - h[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]
 
```

Now, we can upate the parameters  <img src="https://latex.codecogs.com/svg.latex?\mathbf{U}, \mathbf{V}, \mathbf{W}" align = 'center' />.

```
# Performs one step of SGD.
def numpy_sdg_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW
    
```
The final step for training is SGD, where we apply the updated parameters and decrease the loss.

```
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) &gt; 1 and losses[-1][1] &gt; losses[-2][1]):
                learning_rate = learning_rate * 0.5 
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
```

#### TensorFlow Approach

**The TF approach is the main part of this post!!!!**
After we went through the RNN script, it'll be better for us to understand TensorFlow Approach. As we can see, the difficult part of RNN script is the gradients. Especially when we'd like to improve the model by add our own rules. It's really time-costing and hard to calculate the gradients. In TensorFlow, one of the best things I like is that we don't need to calculate the gradients, since the TF could hanble this. But it is not the most important point in TF. In TF, We first construct the graph by define variables and relations between constants or variables. Only after we construct the RNN graph, can we run the whole in a session. The strongly recommended book for beginner is [Getting Started with TensorFlow](https://www.packtpub.com/big-data-and-business-intelligence/getting-started-tensorflow).

##### Clean data

The clean_data steps are basically the same, except I add bucketing padding, and batching.

* **Pad data**

In that RNN script, we process one pair inputs and outputs per step, which is low effeciency. The better way is that we can process several inputs and outputs per step. However, sentences do not have the same length necessarily. That's why we need to pad sentence into the same length by adding 0 for sentences with shorter length.

For example, there are two sentences <img src="https://latex.codecogs.com/svg.latex?[4,5,6]" align ='center' /> and <img src="https://latex.codecogs.com/svg.latex?[6,5,7,9,4]" align ='center' />. After padding, we get the two sentences <img src="https://latex.codecogs.com/svg.latex?[4,5,6,0,0]" align ='center' /> and <img src="https://latex.codecogs.com/svg.latex?[6,5,7,9,4]" align ='center' />.

* **Bucket data**

Next, we have to know how long that we pad to. There are millions of sentences waiting to be processed. Most sentences are short to middle length, while there are always sentence with super many words. We can't pad all sentences into the highest length, which must be time-costing for computing. That's when bucket works. We deliver the sentences into different bucket according to their real length. The sentences in the same bucket will be padded into the same length.

* **Batch data**

All work that has been done above is to batch the sentence so that we can process several sentences per training step.
For sentences <img src="https://latex.codecogs.com/svg.latex?[4,5,6]" align ='center' />, <img src="https://latex.codecogs.com/svg.latex?[5,3,2]" align ='center' />, <img src="https://latex.codecogs.com/svg.latex?[6,5,2,1]" align ='center' />, <img src="https://latex.codecogs.com/svg.latex?[9,7,3,4,7]" align ='center' />, we could pad and batch them to a input matrix, like
<div align = 'center'>
<img src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}4&5&6&0&0\\5&3&2&0&0\\6&5&2&1&0\\9&7&3&4&7\end{bmatrix}" title="\begin{bmatrix}4&5&6&0&0\\5&3&2&0&0\\6&5&2&1&0\\9&7&3&4&7\end{bmatrix}" /></div>

##### TensorFlow functions

To build the RNN model in TensorFlow, there are a few functions that I'd like to introduce

* `tf.nn.embedding_lookup(matrix, index)`

This function is to extract specific 'index' column of the 'matrix'

* `lstm_cell() = tf.contrib.rnn.BasicLSTMCell(hidden_size)` 

We use the LSTM cell here, which could better process the long dependency than normal RNN cell. The related paper is here [LSTM cell](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf). In LSTM cell, we don't simply pass the last hidden state to next hidden state. Instead, we can choose to pass or forget by adding one more parameter, forget gate <img src="https://latex.codecogs.com/svg.latex?f_t" title="f_t" />. Also, we can choose to pass the input data or not by adding parameter <img src="https://latex.codecogs.com/svg.latex?i_t" title="i_t" />. The details are illustrated in paper well.

We can use multiple LSTM layer by `tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(number_of_layers)])`
    
To get the know more cell functions, [here](https://www.tensorflow.org/versions/r0.12/api_docs/python/rnn_cell/)

* `tf.nn.dynamic_rnn` This function has been illustrated well in [dynamic_rnn](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/). We just pass the parameters cell, inputs, sequence\_length,time\_major, dtype into the function and get the states and outputs of the input sentences. Note the outputs are with dimension [batch\_size, max\_time, cell.output\_size], which means it includes all outputs of each time step (word). Also here the outputs are just the hidden states <img src="https://latex.codecogs.com/svg.latex?h_{t}" align = 'center'/>, which have to be multiplied by  <img src="https://latex.codecogs.com/svg.latex?\mathbf{W}" /> to get the  <img src="https://latex.codecogs.com/svg.latex?o_{t}" align = 'center'/>.


* Normalize the gradients

Since we only care about which direction to move and want a more stable training, we can normalize the gradients before applied to optimization. This part of codes could handle it.

```
trainable = tf.trainable_variables()
        
gradients_norms = []
        
updates = []
        
opt = tf.train.GradientDescentOptimizer(learning_rate)   
        
gradients = tf.gradients(loss_ave, trainable)
    
clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    
gradients_norms.append(norm)
    
updates.append(opt.apply_gradients(zip(clipped_gradients, trainable), 
                                       global_step= global_step))
        
```

* Model saver

Since it is easy that we stop the training due to multiple reasons, we'd like to continue to train rather start over. These codes could handle it.

```
saver = tf.train.Saver(tf.global_variables())

saver.restore(sess, os.path.join(ckpt.model_checkpoint_path))
```


#### Implementation

Finally, the implematation!!! Since there are 3 python files calling each other. I will not post here. The repo has been updated, [rnn](https://github.com/Keyspan/exploring_nlp/tree/master/rnn_lstm). You can run the train() function in `rnn_graph.py`.


#### Results

For now, it is still training. But the sample results are like,

'I can finnaly go fishing!!!'. We can use this model to produce sentences.

#### Application

Of course, RNN could be used in any problems. If we use the last hidden state of each sentence to represent each sentence, we embed the sentence with one vector. We can use this vector to classfify this sentence by choosing targets as differnet categories, which has wide application on business problems. The codes only need to be edited slightly and take whatever you need.

Good Luck!!!!! Next post, Seq2Seq Model--Chat Robot!!!

#### Reference

* WildML: [http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
* Backpropagation:[https://www.cs.swarthmore.edu/~meeden/cs81/s10/BackPropDeriv.pdf](https://www.cs.swarthmore.edu/~meeden/cs81/s10/BackPropDeriv.pdf)
* BPTT: [http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)
* BPTT Derivation: [https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf](https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf)
* Getting Started with TensorFlow: [https://www.packtpub.com/big-data-and-business-intelligence/getting-started-tensorflow](https://www.packtpub.com/big-data-and-business-intelligence/getting-started-tensorflow)