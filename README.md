Text Summarization with Seq2Seq, Transformers, and Reinforcement Learning

This repository contains Jupyter notebooks for three different methods of text summarization using the XSum dataset:

1. Sequence-to-Sequence (Seq2Seq) model
2. Transformers (T5) model
3. Reinforcement Learning (T5 with Proximal Policy Optimization)

## 1. Sequence-to-Sequence (Seq2Seq) Model

In this approach, we use a simple Seq2Seq model with LSTM layers to perform text summarization. The model consists of an encoder and a decoder, where the encoder processes the input text, and the decoder generates the summary.

### Preprocessing

The dataset is preprocessed to convert text into sequences of integers. We use Keras' Tokenizer to tokenize the text and pad the sequences to a fixed length.

### Training and Checkpoints

The Seq2Seq model is trained using the preprocessed dataset, with a training and validation split. Model checkpoints are saved during training so that you can load a previously trained model and resume training from a specific epoch if needed.

To load a checkpoint and continue training, you need to:

1. Load the checkpoint using `model.load_weights(checkpoint_path)`.
2. Update the `initial_epoch` argument in the `model.fit()` function to the epoch number you want to start from.
3. Train the model with the updated `initial_epoch` and the desired number of additional epochs.

### Evaluation

The model is evaluated using the test dataset, and the BLEU score is calculated to measure the performance of the model.

## 2. Transformers (T5) Model

In this approach, we use the T5 (Text-to-Text Transfer Transformer) model for text summarization.

### Preprocessing

The XSum dataset is preprocessed using the T5 tokenizer, which tokenizes the input text and converts it into sequences of integers.

### Training

The T5 model is fine-tuned using the preprocessed dataset, with a training and validation split. The model is trained using the Hugging Face Transformers library.

### Evaluation

The model is evaluated using the test dataset, and the BLEU score is calculated to measure the performance of the model.

## 3. Reinforcement Learning (T5 with Proximal Policy Optimization)

In this approach, we use Reinforcement Learning to optimize the T5 model for text summarization.

### Preprocessing

The XSum dataset is preprocessed using the T5 tokenizer, as in the previous approach.

### Training

The T5 model is fine-tuned using Proximal Policy Optimization (PPO). The model receives a reward signal based on the ROUGE score, which measures the overlap between the generated summary and the human-generated summary.

### Evaluation

The model is evaluated using the test dataset, and the BLEU and ROUGE scores are calculated to measure the performance of the model.

## Dependencies

- Python 3.8+
- TensorFlow 2.6+
- Hugging Face Transformers 4.11+
- PyTorch 1.9+
- Datasets 1.12+
- NLTK 3.6+
- Wandb 0.12+

## Usage

To run each of the methods, open the corresponding Jupyter notebook:

- Seq2Seq: seq2seq_summarization.ipynb
- Transformers (T5): t5_summarization.ipynb
- Reinforcement Learning (T5 with PPO): rl_t5_summarization.ipynb

## License

This project is licensed under the MIT License.

