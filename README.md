# cervantes

## Contents
- [Introduction](#introduction)
- [Long short-term memory](#vanillalstm)
- [SeqGAN](#seqGAN)
- [Literary musings](#literary-musings)

## Introduction

Text generation in Spanish based on Don Quijote presents several unique and interesting challenges. This project experiments with a wide variety of neural text generation (NTG) techniques such as supervised learning through character-based Recurrent Neural Networks (RNNs) and reinforcement learning via unsupervised Generative Adversarial Networks or GANs for short (Goodfellow et al., 2014). To what extent of success can the literary style of Cervantes, the most widely read Spanish author in the world, be imitated and learned?

## VanillaLSTM

The first version of the model was designed to be a simple baseline. It consists of one embedding layer, a long short-term memory (LSTM) layer, and a dense layer. The VanillaLSTM is character-based precisely because sentences are generated one character at a time based on probability distributions predicting the next character.

One of the main advantages of using a LSTM over a basic RNN include having memory of words which are not restricted to the short term. In addition to the hidden state, there is a cell state which preserves information longer through the forget and remember modules. Pointwise multiplication by a mask vector is used to forget information while remembering is done through tanh and gating memory to contain only what we want to preserve long-term:

<img src="images/LSTM.png" width="420px" height="160px">

LSTMs are very strong at modelling sequential data, and thus has been frequently used to generate text imitating authors such as Shakespeare. However, very few such models have been tested on other languages with more complicated grammatical rules such as Spanish.

Given the huge size of the don Quijote corpus, there were two primary NLP challenges in creating an effective language model. Notably, Cervantes' masterpiece *El ingenioso hidalgo don Quijote de la Mancha* is in an antiquated form of Spanish (Lathrop 2019). The novel has several phonological and grammatical peculiarities:
- Using past subjunctive where one would expect the conditional
- Using future subjunctive which is now replaced by present indicative
- Employing passive voice and contractions prevalently

Spanish has stricter and more nuanced grammatical rules compared to English. There are **16 distinct tenses** which any Spanish verb can be conjugated into. Thus, the training process to reach high accuracy is undoubtly longer.

Following spanish literary corpi such as MegaLite, a UTF-8 encoding was used to represent the original spanish. This setting ensures that accents and punctuation unique to Spanish are preserved in text generation. The text file was also read using UTF-8 signature, which treats the byte order marks as metadata instead of part of the file's content in order to avoid preprocessing bugs with dictionary key access. 

Here is a **general overview** of the model and its shape:

![example model](images/model.png)

With 5,343,051 trainable parameters and running on 5 epochs, I trained a preliminary LSTM model based on *don Quijote* using an Adam optimizer. The sparse categorical cross entropy loss was reduced from 2.6026 in the first epoch to 1.4291 in the last epoch. 

The first attempt of implementing a LSTM had a fatal flaw in the preprocessing step. Regexes were used to strip punctuation and white space, however that removes grammatical structure from the sentences entirly. A sample sentence from this first model is riddled with words that do not exist and grammatical errors:

<details>
  <summary>Example output:</summary>
  
  >```don Quijote Cara dijo era la capa Parciliente si posaría de dresa ser por pencallero para de Harái yuque exegormambién en la cual me dejando exa yacio dijo dichos Espora que esaba harto que él cura son descuy saliado azóna aquellos que darllera Lazandí con la crietpa.```

</details>

Therefore, regexes were removed and puncutation was preserved. Drastic improvement was observed. The model was capable of citing numerous characters (i.e. Camilla and don Fernando) and using different literary styles such as sonnets. However, there was still significant ambiguity in the sentences and meaning was often obfuscated. For example, here are two sample sentences generated from the input "Dulcinea":

<details>
  <summary>Example output:</summary>
  
  >```Dulcinea que de su risponde, porque el camino como un canse nibes del triendo en cielto, simprino moy de las humándoles, verían y heráspanto.```
  
  > ```Dulcinea dice: Dios hecha hejor donde galer la emparte y mifar se decernas, aunque natura de otras nicús mantas destas de aligaron.```

</details>

These sentences are a significant improvement in several respects. The model is able to employ conjuctions and transition words with correct punctuation such as porque (because) and aunque (although). Moreover, the second sentence seems to be on the verge of meaning (although it is still gibberish), as it starts with a prayer to God (Dios).

This model could have benefitted from significantly more training epochs, however that would be much more computationally taxing in both the framework of time and cost. The question thus is if there is a **more efficient way** to increase accuracy in a shorter time frame.

## SeqGAN

The Sequence Generative Adversarial Network consists of a generator and a discriminator which consists of one dense layer with sigmoid activation. After running 1000 batches, there was a memory allocation error due to the size of the dataset. In order to resolve this issue, the model's training ran on Brown's High Performance Computing Clusters, known as [OSCAR](https://docs.ccv.brown.edu/oscar/).

Here is a general diagram of how SeqGAN operates:

![SeqGAN model](images/seqgan.png)

The Discriminator in the SeqGAN model uses pooling and applies dropout regularization for better generalization. Its metric is softmax cross-entropy loss with L2 regularization and is optimized with the Adams optimizer. The Generator creates sequences of tokens through a combination of an embedding layer, recurrent unit, and output unit. It then uses the Adam optimizer for training and applies gradient clipping. The Rollout class implements a rollout policy, a technique used in reinforcement learning to improve the performance of the Generator model by fine-tuning its output based on the feedback received from the Discriminator. With this policy, rewards are calculated to update the Generator's parameters.

With 120 pre-train epochs, the model pre-trained both the generator and discriminator, with cross-entropy loss being reduced from 10.216 to 8.267. SeqGAN was significantly faster in running more epochs efficiently, however training also took longer because multiple models had to be trained as opposed to one.

But, in the context of the Spanish language, SeqGAN proved to be superior since the discriminator was able to root out and correct the grammatical errors of the generator. The generator, through reinforcement learning rewards, was even able to learn the antiquated grammar of Cervantes' time. Given that LSTMs were developed with an anglocentric focus in mind, that is tackling natural language problems in English, Spanish provided a harder challenge which 1. the LSTM struggled with and 2. SeqGAN exceled with. Reinforcement Learning thus provides a promising paradigm which offers huge versatility and adaptability, but requires more optimization in terms of resource costs.

## Literary musings
Cervantes famously writes in the first part of *Quijote* that the author of the novel is a *morisco* historian by the name of Cide Hamete Benengeli. However, we must be cautious in using the word 'novel'. The term and its connotation today had yet to be invented back then. 

The contemporaries of Cervantes would instead refer to this adventure as a *historia*, a history. The narrator goes to great lengths to assert the most truthful nature of the story, and curiously, there are even chapters which Cide Hamete identifies as apocryphal, that is being fake (i.e. the Cave of Montesinos and the conversation between Sancho and his wife Teresa). But, all history pretends to be history (see Wardropper). 

One of the greatest characteristics of *el Quijote* is its inability to be defined. It escapes the realm of facts. Is the real name of don Quijote Alonso Quijano? We will never know. Great linguistic uncertainty pervades *el Quijote*. Sancho's wife is referred to as Teresa, Juana, Maria and even the surname of the titular character don Quijote is uncertain, sometimes being Quijada, Quesada, or Quijana. In criticizing the shortcomings of writing, as being incapable of encapsulating truth, Cervantes weaves a great tapestry of human expression and subjectivity. 

Writing is rigid. Once text is put on a page, it dies and is unable to defend itself (see Plato' attack of writing in the *Phaedrus* and his *Seventh Letter*). With the advent of large language models (LLMs) and transformers (see the paper "Attention Is All You Need"), we are once again forced to reevaluate writing in its essence. 

Certainly from the perspective of Cervantes, authenticity in writing is overrated. Most writing functionally is a means, a medium to convey information. The best type of writing, however, is one that forces the reader to think for themselves, to learn from what is left unsaid. LLMs are not yet at the level of Cervantes, who lived almost half a millenia ago. His innovation, the first modern novel, remains a towering literary achievement. 

But, Cervantes does give us a challenge. He ends the first part of *el Quijote* with a line from Ariosto, a famous Italian poet:

*Forsi altro canterà con miglior plectio.* 

In English, this translates to "perhaps another will sing with a better pluck". Akin to a musician, the author so too is singing a unique ode. Cervantes may have finished writing the story of don Quijote, but he anticipates it will be sung again. Indeed, his tale has been told countless times in other shapes and forms, inspiring *Madame Bovary* and *The Idiot*. 

Human creativity is irreplacable. Poetic madness drives don Quijote on his famous adventures and literature to greater heights. Don Quijote overflows his own humanity, his own capacities, exceeding the bounds of human behavior. Cervantes too lives on through this history, and challenges us to do the same.