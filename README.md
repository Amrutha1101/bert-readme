# Team ID : 
SC24

## Team: 
[C K Amrutha](https://github.com/Amrutha1101) (1CR19CS032) \
[Dhruthi H H](https://github.com/Dhruthihh) (1CR19CS045)  \
[G Sahana](https://github.com/Gondhi-Sahana) (1CR19CS063)  \
[Dr G Radhakrishnan](https://github.com/?) 


## Abstract 
Question & Answering (Q&A) systems can have a huge impact on the way information is accessed in today's world. In the domain of computer science, Q&A lies at the intersection of Information Retrieval and Natural Language Processing.

It deals with building intelligent systems that can provide answers, for user generated queries, in a natural language.

We are developing Question & Answering system using DistilBERT which is a variant of BERT where user can give dataset and ask questions to our model which will then give answers accordingly and also have evaluated our model and the f1 score is 88% and exact match to be 81%.

## Model description

BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way 
(which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. 
More precisely, it was pretrained with two objectives:

Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.

Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not.

This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard classifier using the features produced by the BERT model as inputs.


## Evaluation results

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on a squad dataset.
It achieves the following results on the evaluation set:
- Train Loss: 1.2665
- Epoch: 3

{'exact_match': 81.18259224219489, 'f1': 88.67381321905516} is achieved which is greater than existing QA models.

## Training hyperparameters
learning_rate: 2e-5,
num_train_epochs = 3,
batch_size=16

## Training results

| Train Loss | Epoch |
|:----------:|:-----:|
| 1.2665     | 3   |


## Framework versions

- Transformers 4.28.1
- TensorFlow 2.12.0
- Datasets 2.12.0
- Tokenizers 0.13.3

## ⚙️ Instructions
We trained the model on [colab](https://colab.research.google.com/drive/1EywpKLcbT4irm9ORFPl9CW_PymQ3hZPP#scrollTo=R0ITsFRBf6Kf) which is easily accessible. Make sure that you have logged in to colab notebook and set the runtime type to GPU to run the model.
You will require to create an account on [hugging-face](https://huggingface.co/) and generate a write token which is also asked in colab's 2nd cell so as to connect to hugging face for frontend, hugging face notebook.

The [interface](https://huggingface.co/GSahana/bert-finetuned-squad) is hosted on huggingface mode hub
