import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
import math

# Transcript is one line, so we select it and change question mark for dots so that we split it correctly.
def split_sentence(text):
    text = text.replace("?", ".")
    sentences = text.split('. ')
    sentences[-1] = sentences[-1].replace('.', '')
    return sentences

def unify_sentence(sentences):
    # Get the length of each sentence
    sentece_length = [len(each) for each in sentences]
    # Determine longest outlier
    long = np.mean(sentece_length) + np.std(sentece_length) *2
    # Determine shortest outlier
    short = np.mean(sentece_length) - np.std(sentece_length) *2
    # Shorten long sentences
    text = ''
    prev_each = ''

    for i, each in enumerate(sentences):
        if each == prev_each or len(each.strip()) == 0:
            continue
        if len(each) > long:
            # let's replace all the commas with dots
            comma_splitted = each.replace(',', '.')
            text+= f'{comma_splitted}. '
        else:
            text+= f'{each}. '

        prev_each = each
    
    sentences = text.split('. ')
    sentences[-1] = sentences[-1].replace('.', '')
    # Now let's concatenate short ones
    text = ''
    for each in sentences:
        if len(each) == 0:
            continue
        if len(each) < short:
            text+= f'{each} '
        else:
            text+= f'{each}. '
    
    return text

def rev_sigmoid(x:float)->float:
    return (1 / (1 + math.exp(0.5*x)))
    
def activate_similarities(similarities:np.array, p_size=10, order=5)->np.array:
    """ Function returns list of weighted sums of activated sentence similarities
    Args:
        similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
        p_size (int): number of sentences are used to calculate weighted sum 
    Returns:
        list: list of weighted sums
    """
    x = np.linspace(-10,10,p_size)
    # Then we need to apply activation function to the created space
    y = np.vectorize(rev_sigmoid) 
    # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
    activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
    ### 1. Take each diagonal to the right of the main diagonal
    diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
    ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
    diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
    ### 3. Stack those diagonals into new matrix
    diagonals = np.stack(diagonals)
    ### 4. Apply activation weights to each row. Multiply similarities with our activation.
    diagonals = diagonals * activation_weights.reshape(-1,1)
    ### 5. Calculate the weighted sum of activated similarities
    activated_similarities = np.sum(diagonals, axis=0)
    ### 6. Find relative minima of our vector. For all local minimas and save them to variable with argrelextrema function
    minmimas = argrelextrema(activated_similarities, np.less, order=order) #order parameter controls how frequent should be splits. I would not reccomend changing this parameter.

    return minmimas

def correct_chunks(chunks):
    prev_chunk = None
    new_chunks = []
    for chunk in chunks:
        if prev_chunk:
            chunk['text'] = prev_chunk['text'] + chunk['text']
            chunk['timestamp'] = (prev_chunk['timestamp'][0], chunk['timestamp'][1])

        if not chunk['text'].endswith('.'):
            prev_chunk = chunk
        else:
            new_chunks.append(chunk)
            prev_chunk = None
    return new_chunks

def gen_parag(input_chunks, model_name='all-minilm-l6-v2', p_size=10, order=5):
    sentences_all = []
    timestamps_all = []
    
    corrected_chunks = correct_chunks(input_chunks)
    
    for chunk in corrected_chunks:
        sentences = split_sentence(chunk['text'])
        text = unify_sentence(sentences)
        text = text.strip()
        sentences = text.split('. ')
        sentences[-1] = sentences[-1].replace('.', '')
        timestamps = [chunk['timestamp']]*len(sentences)

        sentences_all += sentences
        timestamps_all += timestamps
    
    # Embed sentences
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences_all)
    # Create similarities matrix
    similarities = cosine_similarity(embeddings)

    # Let's apply our function. For long sentences i reccomend to use 10 or more sentences
    minmimas = activate_similarities(similarities, p_size=p_size, order=order)

    # Create empty string
    split_points = [each for each in minmimas[0]]
    text = ''

    para_chunks = []
    para_timestamp = []
    start_timestamp = 0
    
    for num, each in enumerate(sentences_all):
        current_timestamp = timestamps_all[num]
        
        if text == '' and (start_timestamp == current_timestamp[1]):
            start_timestamp = current_timestamp[0]
        
        if num in split_points:
            #text+=f'{each}. '
            para_chunks.append(text)
            para_timestamp.append([start_timestamp, current_timestamp[1]])
            text = f'{each}. '
            start_timestamp = current_timestamp[1]
        else:
            text+=f'{each}. '

    if len(text):
        para_chunks.append(text)
        para_timestamp.append([start_timestamp, timestamps_all[-1][1]])
    
    return para_chunks, para_timestamp
