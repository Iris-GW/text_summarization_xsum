# %%
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import wandb
from wandb.keras import WandbCallback
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.callbacks import ModelCheckpoint

# Download the Xsum dataset
dataset = load_dataset('xsum')


# %%
# Initialize wandb
wandb.init(project="text_summarization", name="seq2seq_summarization_e")


# %%
# Preprocess the data
nltk.download('punkt')
stemmer = PorterStemmer()

def preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Stem the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join the stemmed tokens back into a single string
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return preprocessed_text

preprocessed_data = []
for example in dataset['train']:
    article = example['document']
    summary = example['summary']
    preprocessed_article = preprocess(article)
    preprocessed_summary = preprocess(summary)
    preprocessed_data.append((preprocessed_article, preprocessed_summary))


# %%
# Reduce dataset size
train_data, _ = train_test_split(preprocessed_data, test_size=0.5, random_state=42)

# %%
# Create tokenizer and fit on texts
encoder_inputs_train = [example[0] for example in train_data]
decoder_inputs_train = ['<start> ' + example[1] for example in train_data]
decoder_outputs_train = [example[1] + ' <end>' for example in train_data]

tokenizer = Tokenizer(filters='', lower=False, split=' ')
tokenizer.fit_on_texts(encoder_inputs_train + decoder_inputs_train + decoder_outputs_train)

encoder_inputs_train = tokenizer.texts_to_sequences(encoder_inputs_train)
decoder_inputs_train = tokenizer.texts_to_sequences(decoder_inputs_train)
decoder_outputs_train = tokenizer.texts_to_sequences(decoder_outputs_train)

vocab_size = len(tokenizer.word_index) + 1
input_vocab_size = vocab_size
target_vocab_size = vocab_size

# Set the model configuration
config = wandb.config
config.input_vocab_size = input_vocab_size
config.target_vocab_size = target_vocab_size
config.embedding_dim = 128
config.lstm_units = 256
config.batch_size = 16
config.epochs = 10

# Set the maximum length for the encoder and decoder inputs
max_encoder_length = 150
max_decoder_length = 150

# Pad or truncate the sequences to the desired length
encoder_inputs_train = pad_sequences(encoder_inputs_train, maxlen=max_encoder_length, padding='post', truncating='post')
decoder_inputs_train = pad_sequences(decoder_inputs_train, maxlen=max_decoder_length, padding='post', truncating='post')
decoder_outputs_train = pad_sequences(decoder_outputs_train, maxlen=max_decoder_length, padding='post', truncating='post')

# Reshape the decoder_outputs_train
decoder_outputs_train = np.expand_dims(decoder_outputs_train, -1)

# Define the layers
encoder_inputs = Input(shape=(None,))
decoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=input_vocab_size, output_dim=128)(encoder_inputs)
decoder_embedding = Embedding(input_dim=target_vocab_size, output_dim=128)(decoder_inputs)

encoder_outputs, state_h, state_c = LSTM(256, return_sequences=True, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_outputs, _, _ = LSTM(256, return_sequences=True, return_state=True)(decoder_embedding, initial_state=encoder_states)

attention = Attention()([encoder_outputs, decoder_outputs])
x = Concatenate(axis=2)([decoder_outputs, attention])

decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(x)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

print("encoder_inputs_train shape:", encoder_inputs_train.shape)
print("decoder_inputs_train shape:", decoder_inputs_train.shape)
print("decoder_outputs_train shape:", decoder_outputs_train.shape)


# %%
# Define the checkpoint directory
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Create a checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch:03d}.hdf5'),
    save_weights_only=True,
    save_best_only=False,
    verbose=1,
    save_freq='epoch'  # Save after each epoch
)


# %%
model.fit([encoder_inputs_train, decoder_inputs_train],
          decoder_outputs_train,
          batch_size=config.batch_size,
          initial_epoch=9,
          epochs=10,
          callbacks=[WandbCallback(), checkpoint_callback])

# %%
# Load the checkpoint
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch_010.hdf5")
model.load_weights(checkpoint_path)

# %%
from tqdm import tqdm  # import tqdm library
import json

# Create a function to generate summaries using the trained seq2seq model
def generate_summary_seq2seq(article):
    encoder_input = pad_sequences(tokenizer.texts_to_sequences([article]), maxlen=max_encoder_length, padding='post', truncating='post')
    decoder_input = np.zeros(shape=(1, max_decoder_length), dtype='int32')
    decoder_input[0, 0] = tokenizer.word_index['<start>']
    
    summary_generated = []
    for _ in range(max_decoder_length - 1):
        decoder_output = model.predict([encoder_input, decoder_input])
        word_idx = np.argmax(decoder_output[0, -1, :])
        
        if word_idx == tokenizer.word_index['<end>']:
            break
            
        word = tokenizer.index_word.get(word_idx)
        if word is not None:
            summary_generated.append(word)
        
        decoder_input[0, 1:] = decoder_input[0, :-1]
        decoder_input[0, 0] = word_idx
        
    return ' '.join(summary_generated)


# Define a function to evaluate the model using Rouge and BLEU scores
def evaluate_model(generate_summary_function):
    rouge_scores = []
    bleu_scores = []
    rouge = Rouge()
    
    skipped_count = 0
    
    total_examples = len(dataset['test'])
    for i, example in tqdm(enumerate(dataset['test']), total=total_examples):  # use tqdm to show progress
        article = example["document"]
        summary = generate_summary_function(article)
        
        # Skip evaluation if the generated summary is empty
        if not summary:
            skipped_count += 1
            continue
        
        reference = example["summary"]
        rouge_score = rouge.get_scores(summary, reference, avg=True)
        rouge_scores.append(rouge_score)
        
        bleu_score = sentence_bleu([reference.split()], summary.split())
        bleu_scores.append(bleu_score)
        
        # Log the current example being processed
        wandb.log({"eval_example": i+1, "total_eval_examples": total_examples})
    
    # Log the number of skipped examples
    wandb.log({"skipped_eval_examples": skipped_count, "total_eval_examples": total_examples})
    
    return np.mean(rouge_scores), np.mean(bleu_scores)


# Evaluate the seq2seq model
seq2seq_rouge_scores, seq2seq_bleu_scores = evaluate_model(generate_summary_seq2seq)
print("Seq2Seq Rouge Scores:", seq2seq_rouge_scores)
print("Seq2Seq BLEU Scores:", seq2seq_bleu_scores)


# Log Rouge and BLEU scores to WandB
wandb.log({"rouge1": seq2seq_rouge_scores['rouge1'].mid.fmeasure,
           "rouge2": seq2seq_rouge_scores['rouge2'].mid.fmeasure,
           "rougeL": seq2seq_rouge_scores['rougeL'].mid.fmeasure,
           "avg_bleu": seq2seq_bleu_scores})

# Finish the run
wandb.finish()



