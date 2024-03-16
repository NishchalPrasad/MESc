#For running in multiple gpus
from accelerate import Accelerator
from accelerate.logging import get_logger

import deepspeed
from deepspeed.accelerator import get_accelerator

import os
import random
import pandas as pd
import csv
import json 
import time
import datetime
from collections import Counter
import argparse
#import progressbar
#from tensorflow.keras import keras
from tqdm import tqdm
import pandas as pd
#load libraries
import textwrap
#import progressbar
#from tensorflow.keras import keras
import datasets
#from datasets import load_dataset #does not work in pytorch module
#from keras_preprocessing.sequence import pad_sequences #does not work in pytorch module
import numpy as np
import transformers
from transformers import AutoTokenizer,EvalPrediction# TFAutoModel, 
from transformers import AutoModelForSequenceClassification, AutoModel, GPTNeoForSequenceClassification
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split #does not work in pytorch module
from sklearn.metrics import multilabel_confusion_matrix, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer,LabelBinarizer
#import tensorflow as tf 
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch #works in pytorch module
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from ast import literal_eval #works

def pad_sequences(
    sequences,
    maxlen=None,
    dtype="int32",
    padding="pre",
    truncating="pre",
    value=0.0,
):
    """Pads sequences to the same length.
    This function transforms a list (of length `num_samples`)
    of sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence in the list.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` until they are `num_timesteps` long.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding or removing values from the beginning of the sequence is the
    default.
    >>> sequence = [[1], [2, 3], [4, 5, 6]]
    >>> tf.keras.preprocessing.sequence.pad_sequences(sequence)
    array([[0, 0, 1],
           [0, 2, 3],
           [4, 5, 6]], dtype=int32)
    >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, value=-1)
    array([[-1, -1,  1],
           [-1,  2,  3],
           [ 4,  5,  6]], dtype=int32)
    >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post')
    array([[1, 0, 0],
           [2, 3, 0],
           [4, 5, 6]], dtype=int32)
    >>> tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=2)
    array([[0, 1],
           [2, 3],
           [5, 6]], dtype=int32)
    Args:
        sequences: List of sequences (each sequence is a list of integers).
        maxlen: Optional Int, maximum length of all sequences. If not provided,
            sequences will be padded to the length of the longest individual
            sequence.
        dtype: (Optional, defaults to `"int32"`). Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, "pre" or "post" (optional, defaults to `"pre"`):
            pad either before or after each sequence.
        truncating: String, "pre" or "post" (optional, defaults to `"pre"`):
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value. (Optional, defaults to 0.)
    Returns:
        Numpy array with shape `(len(sequences), maxlen)`
    Raises:
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError as e:
            raise ValueError(
                "`sequences` must be a list of iterables. "
                f"Found non-iterable: {str(x)}"
            ) from e

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
        dtype, np.unicode_
    )
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError(
            f"`dtype` {dtype} is not compatible with `value`'s type: "
            f"{type(value)}\nYou should set `dtype=object` for variable length "
            "strings."
        )

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at "
                f"position {idx} is different from expected shape "
                f"{sample_shape}"
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc) :] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x


#------------------------------------------

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Extracting CLS embedding step: LLM (GPT-Neo, GPT-J)")
    
    parser.add_argument(
        "--maxlen",
        type=int,
        default=512,
        help = "The maximum length of the input to the model. 512 for BERT based, 2048 for GPT-Neo, GPT-J.",
    )
    
    parser.add_argument(
        "--length",
        type=int,
        default=510,
        help = "the length of the input chunk",
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help = "Amount of overlap (in number of 'tokens') between adjacent chunks",
    )
    parser.add_argument(
        "--combine_layers",
        type=bool,
        default=True,
        help = "To extract CLS embeds from only the last layer (False), or from last few layers(True)",
    )
    
    parser.add_argument(
        "--layers_from_end",
        type=int,
        default=4,
        help = "If --combine layer is True then how many layers to combine?",
    )
    
    parser.add_argument(
        "--loading_model_path",
        type=str,
        default=None,
        help = "The path to load the model from.",
    )
    parser.add_argument(
        "--cuda_number",
        type=int,
        default = 0,
        help = "Specify cuda device if more are available(else two or more simultaneous run of this script with run in the same cuda device and overflow the memory causing OOM error)"
    )
    parser.add_argument(
        "--strat",
        type=int,
        default=None,
        help="strat: 1 (for 512 input length), and 0 (for 2048 input length)",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default=None,
        help="The name of the dataset to use (via the extracted embeddings).",
    )
    
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default='LEGAL-P_E/SIGIR_experiments/finetuned_models/', 
        help="Directory to save the training data in.",
        #required=True,
    )
    
    parser.add_argument(
        "--hggfc_model_name",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--get_train_data",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--get_validation_data",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--get_test_data",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--trained_with_deepspeed_accelerate",
        type=bool,
        default=False,
        help="Trained with deepspeed in accelerate"
    )
    parser.add_argument(
        "--trained_with_accelerate",
        type=bool,
        default=False,
        help="trained only with accelerate"
    )
    parser.add_argument(
        "--trained_without_accelerate",
        type=bool,
        default=False,
        help="trained without distributed computing (accelerate)"
    )
    parser.add_argument(
        "--extracting_from_ft",
        type=bool,
        default=True,
        help = "whether extracting from a finetuned model?"
    )
    parser.add_argument(
        "--get_explanation_data",
        type=bool,
        default=False,
        help="extract CLS embeddings for explanation"
    )
    parser.add_argument(
        "--path_train_dat",
        type=str,
        default="LEGAL-P_E/SIGIR_experiments/finetuned_models/",
        help="path to save the ectracted CLS embeddings"
    )
    parser.add_argument(
        "--which_split",
        type=int,
        default=1,
        help="Which ILDC split (out of three) should be be parsed? (keep track of thw split number in the file name)"
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_subset is None:
        raise ValueError("Need a dataset name.")
    if args.dataset_subset in ["ildc"]:
        args.get_explanation_data = True
        print("since the max run time in jean-zay is 20 hours and parsing through ILDC (with A100 80GB GPU) takes more than 20 hours, we split the dataset and parse in three steps. The step number can be found in the output and also in the file name. Please use that number in the argument input (--which_split) for parsing the remaining data. The first pass of this script for ILDC will inherently parse through the development and test dataset (you can switch thet off), so you need not pass those arguments for the remaining splits.")
        print(f"\n\nParsing spli {args.which_split} of 3 for {args.dataset_subset}'s train set.\n\n")

    return args
    
def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of original labels."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)

def convert_to_binarised_labels(data, column: str, vocabulary: list = None, lookup = None, return_lookup: bool = False, problem_type: str = "multi_class"): #or multi_label
    #labels = tf.ragged.constant(train_data[column].values)
    #if lookup == None:
        #if vocabulary:
        #    lookup = tf.keras.layers.IntegerLookup(output_mode="multi_hot", vocabulary = vocabulary, vocabulary_dtype="int64")
        #else:
        #    lookup = tf.keras.layers.IntegerLookup(output_mode="multi_hot")
        #    lookup.adapt(labels)
    #    lookup = MultiLabelBinarizer()
    #    labels_binarized = lookup.fit_transform(train_data[column])
    #labels_binarized = lookup(labels).numpy().astype('int32')
    #else:
    if problem_type == "multi_class":
        data['labels'] = np.zeros((len(data),), dtype = object)
        if lookup == None:
            lookup_ = LabelBinarizer()
    else:
        if lookup == None:
            lookup_ = MultiLabelBinarizer()
    if lookup == None:
        lookup_.fit(data[column]) #data[column]
        labels_binarized = lookup_.transform(data[column])
    else:
        labels_binarized = lookup.transform(data[column])
    #print(labels_binarized)

    #print(lookup_.classes_)
    for i in tqdm(range(len(data))):
        data['labels'][i] = labels_binarized[i]
    if return_lookup: 
        return data, lookup_
    else:
        return data
def concatenate_list_of_texts(data, column: str):
    for i in range(len(data)):
        conc_text = ""
        for j in range(len(data[column][i])):
            conc_text+=data[column][i][j][4:]
        data[column][i] = conc_text
    return data
def get_vocabulary(df, column, problem_type: str = 'multi_label'):
    vocabulary = []
    if problem_type == 'multi_label':
        for doc in range(len(df)):
            labels = df[column][doc]
            for label in labels:
                if label not in vocabulary:
                    vocabulary.append(label)
    elif problem_type == 'multi_class':
        for doc in range(len(df)):
            label = df[column][doc]
            if label not in vocabulary:
                vocabulary.append(label)
    vocabulary.sort()
    return vocabulary

#trying out chunk level embedding
def generate_attention_mask(input_ids):
  attention_masks = []
  for id in input_ids:
    masks = [int(token_id > 0) for token_id in id]
    attention_masks.append(masks)
  return attention_masks
  
def grouped_input_ids(tokens, tokenizer, length: int, overlap: int, maxlen: int, indexing: bool = True):
  tokenised_chunks = []
  seq_lengths = []
  left = 0
  right = length
  indexer = 0
  while(left<len(tokens)):
    tokenised_chunks.append(tokens[left:min(right,len(tokens))])
    left+=length-overlap #100 overlap
    right+=length-overlap
    indexer+=1
  
  CLS = tokenizer.bos_token#cls_token
  SEP = tokenizer.eos_token#sep_token
  pad_value = tokenizer.convert_tokens_to_ids(SEP)
  encoded_chunks = []
  for t in tokenised_chunks:
    t = [CLS] + t + [SEP]
    #print(t)
    enc_chunk = tokenizer.convert_tokens_to_ids(t)
    encoded_chunks.append(enc_chunk)
    seq_lengths.append(len(t))
  encoded_chunks = pad_sequences(encoded_chunks, maxlen=maxlen, value = pad_value, dtype='long', padding='post')
  attention_mask = generate_attention_mask(encoded_chunks) 
  if indexing:
    return encoded_chunks, attention_mask, seq_lengths, indexer
  else:
    return encoded_chunks, attention_mask, seq_lengths, None

def get_output_for_one_vec(model, device, input_id, att_mask, seq_lengths, combine_layers:bool = True, layers_from_end: int = 4, last_token: int = 1):
  input_ids = torch.tensor(input_id)
  att_masks = torch.tensor(att_mask)
  input_ids = input_ids.unsqueeze(0)
  att_masks = att_masks.unsqueeze(0)
  #model.eval()
  input_ids = input_ids.to(device)
  att_masks = att_masks.to(device)
  with torch.no_grad():
    encoded_layers = model(input_ids=input_ids, token_type_ids=None, attention_mask=att_masks, output_hidden_states=True, return_dict = True)
    #print(encoded_layers.sequence_lengths)
    #print(f"fencoded_layers{encoded_layers}")
    #print(f"encoded_layers['hidden_states'][33].shape:{encoded_layers['hidden_states'][33].detach().cpu().numpy().shape()}")
    #print(f"encoded_layers['hidden_states']'s length = {len(encoded_layers['hidden_states'])}")
  #print(len(encoded_layers[1][12]))
  num_layers = len(encoded_layers['hidden_states']) - 1
  if combine_layers:
    vec = []
    for i in range(layers_from_end):
      vec.append(encoded_layers['hidden_states'][num_layers-i][0][seq_lengths-last_token].detach().cpu().numpy()) #seq_lengths-last_token
      #print(encoded_layers['hidden_states'][num_layers-i][0].detach().cpu().numpy().shape)
    return vec
  else:
  #vec = encoded_layers#[1][12][0][-1]
  #vec = vec.detach().cpu().numpy()
    vec = encoded_layers[1][num_layers-i][0][seq_lengths-last_token]
    vec = vec.detach().cpu().numpy()
    return vec

def generate_training_data(df,model, device, tokenizer, text_column:str, label_column:str, remove_first: bool = False, length: int = 510, overlap: int = 100, maxlen = 512, indexing: bool = True, combine_layers: bool = True, layers_from_end: int = 4):
    all_docs, labels, indexes = [], [], []
    for i in tqdm(range(len(df[text_column]))):
        texts = df[text_column].iloc[i]
        tokens = tokenizer.tokenize(texts)
        document_label = df[label_column].iloc[i]
        labels.append(document_label)

        if remove_first:
            if(len(toks) > 10000):
                toks = toks[len(toks)-10000:]

        _input_ids, _attention_masks, seq_lengths, index = grouped_input_ids(tokens, 
                                                                tokenizer, 
                                                                length = length, 
                                                                overlap = overlap, 
                                                                maxlen = maxlen, 
                                                                indexing = indexing)
        #print(seq_lengths)
        #print("length mask:{}".format(len(_attention_masks)))
        #print("length input ids: {}".format(len(_input_ids)))
        vecs = []
        for index,(ii, seq_len) in enumerate(zip(_input_ids,seq_lengths)):
            vecs.append(get_output_for_one_vec(model, device, ii, _attention_masks[index], seq_lengths=seq_len, combine_layers = combine_layers, layers_from_end = layers_from_end, last_token = 2))
        one_doc = np.asarray(vecs)
        all_docs.append(one_doc)
        indexes.append(index)
    
    all_docs = np.asarray(all_docs, dtype=object)
    labels = np.asarray(labels, dtype=object)
    indexes = np.asarray(indexes, dtype=object)
        
    if indexing:
        return all_docs, labels, indexes
    else:
        return all_docs, labels, None
#------------------------------------------Parameters-------------------------------------------------------------------------------------------------------------------------

def main():
    
    args = parse_args()
    dataset_subset = args.dataset_subset
    strat = args.strat
    
    print("Extraction started!\n")
    data_sub_strategy = 0
    #"if oversample or not after combining 'multi' and 'single' dataset"
    hggfc_model_name_=args.hggfc_model_name.replace("/","_")
    
    print(f"\nExtraction for {args.dataset_subset} started!")
    #dataset_subset = "scotus"

    """
    path_train_dat = args.save_dir+f"{args.dataset_subset}/{hggfc_model_name_}/Strategy_{args.strat}/sub_strategy_{data_sub_strategy}/Extracted_CLS_data_{hggfc_model_name_}/chunks_with_{args.overlap}_overlap_and_{args.maxlen}_input-length/"
    """ 
    path_train_dat = args.path_train_dat
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    accelerator = Accelerator()       
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #if os.path.isdir(path_train_dat)==False:
    #    os.makedirs(path_train_dat)
        
    if args.dataset_subset in ["eurlex","ecthr_a","ecthr_b"]:
        type_of_classification = "multi_label"#"multi_class"
        label_ = 'labels'
        if args.dataset_subset in ["ecthr_a","ecthr_b"]:
            id2label = {idx:label for idx, label in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
            label2id = {label:idx for idx, label in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
    elif args.dataset_subset in ["scotus"]:
        type_of_classification = "multi_class"
        label_ = 'labels'
    elif args.dataset_subset in ["ildc"]:
        type_of_classification = "binary"
        label_ = 'label'
        
        
    vocabulary = []
    if args.dataset_subset in ["eurlex","ecthr_a","ecthr_b","scotus"]:
        remove_first = False
        #load dataset 
        dataset = datasets.load_from_disk(f"/gpfsdswork/dataset/HuggingFace/lex_glue/{args.dataset_subset}")
        #dataset = datasets.load_dataset("lex_glue", args.dataset_subset)

        train_data = dataset['train'].to_pandas()
        validation_data = dataset['validation'].to_pandas()
        test_data = dataset['test'].to_pandas()
        text_column = train_data.columns[0]
        label_column = train_data.columns[1]
        print("\nDataset's slice view:\n")
        print(train_data)
        """
        for doc in range(len(train_data)):
            labels = train_data['labels'][doc]
            for label in labels:
                if label not in vocabulary:
                    vocabulary.append(label)
        vocabulary.sort()
        """

        vocabulary = get_vocabulary(train_data, label_column, problem_type = type_of_classification)
        
        train_data, lookup = convert_to_binarised_labels(train_data, label_column, vocabulary=vocabulary, return_lookup = True, problem_type = type_of_classification)
        validation_data = convert_to_binarised_labels(validation_data, label_column, vocabulary=vocabulary, lookup = lookup, problem_type = type_of_classification)#, lookup = lookup)
        test_data = convert_to_binarised_labels(test_data, label_column, vocabulary=vocabulary, lookup = lookup, problem_type = type_of_classification)

        #train_data = convert_to_binarised_labels(train_data, label_, return_lookup = False)
        #validation_data = convert_to_binarised_labels(validation_data, label_)#, lookup = lookup)
        #test_data = convert_to_binarised_labels(test_data, label_)#, lookup = lookup)
        if type_of_classification == "multi_label":
            train_data = concatenate_list_of_texts(train_data, column = "text")
            validation_data = concatenate_list_of_texts(validation_data, column = "text")
            test_data = concatenate_list_of_texts(test_data, column = "text")
        del dataset
        
        split_name = ""
        
    elif args.dataset_subset in ["ildc"]:
        remove_first = True
        #Edit this is block
        data = pd.read_csv(r"LEGAL-PE/SIGIR_experiments/datasets/ILDC/ILDC_multi.csv")
        data_s = pd.read_csv(r"LEGAL-PE/SIGIR_experiments/datasets/ILDC/ILDC_single.csv")
        train_data = data.query("split=='train'")
        train_data_s = data_s.query("split=='train'")
        validation_data = data.query("split=='dev'")
        test_data = data.query("split=='test'")
        exp_data = pd.read_csv(r"LEGAL-PE/SIGIR_experiments/datasets/ILDC/anno_dataset.csv")
        exp_data.rename(columns={'text_y':'text'}, inplace=True)
        
        train_data = train_data.append(train_data_s, ignore_index=True)
        del train_data_s, data_s, data
        label_column = label_
        text_column = 'text'
        
        random.seed(100)
        train_data_0, train_data_1, train_data_2 = np.array_split(train_data, 3)
        
        if args.which_split == 1:
            train_data = train_data_0
        elif args.which_split == 2:
            train_data = train_data_1
        elif args.which_split == 3:
            train_data = train_data_2
        print(train_data)
        split_name = f"{args.which_split}_of_3"
    
    
    #-----------------------------------------------------------------------------------
    #hggfc_model_name = 'EleutherAI/gpt-j-6B'
    print(f"\nLoading {args.hggfc_model_name} model and tokenizer")
    model_path = "/gpfsdswork/dataset/HuggingFace_Models/"+args.hggfc_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #if tokenizer.pad_token is None:
    #    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.special_tokens_map)
    print("\nTokenizer Loaded")
    
    if type_of_classification == "multi_label":   
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            #torch_dtype=torch.float16,
            problem_type="multi_label_classification", 
            num_labels=len(vocabulary),
            id2label=id2label,
            label2id=label2id
        )
    #for multi-class classification
    elif type_of_classification == "multi_class":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            #torch_dtype=torch.float16,
            num_labels=len(vocabulary)
        )
        
    elif type_of_classification == "binary":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            #torch_dtype=torch.float16,
            #ignore_mismatched_sizes = True
        )
 
    
    
    """
    --extracting_from_ft
    --trained_without_accelerate
    --trained_with_accelerate
    --trained_with_deepspeed_accelerate
    --testing_model_path = loading_model_path
    """
    if args.extracting_from_ft:
        if args.trained_without_accelerate:
            checkpoint = torch.load(args.loading_model_path)
            model.load_state_dict(checkpoint['model'])
            del checkpoint
            #model = accelerator.prepare(model)

        elif args.trained_with_accelerate:
            #For models saved as pytorch checkpoints as fp32 to load from state_dict
            model.load_state_dict(torch.load(args.loading_model_path))
            #model = accelerator.prepare(model)
            #accelerator.load_state(args.loading_model_path)
        elif args.trained_with_deepspeed_accelerate:
            model.load_state_dict(torch.load(args.loading_model_path))
                    #dummy_dataloader = validation_dataloader
                    #model, dummy_dataloader = accelerator.prepare(model, dummy_dataloader)
                    #model = accelerator.unwrap_model(model)
                    #model = load_state_dict_from_zero_checkpoint(model, args.loading_model_path)
            #model.load_state_dict(get_fp32_state_dict_from_zero_checkpoint(args.loading_model_path))
            #model = accelerator.prepare(model)

        _name = "with_ft_"

    else:
        print("evaluating on initial pretrained model")
        _name = "without_ft_"
        
    model = accelerator.prepare(model)
    device = accelerator.device
    #device = torch.device(f'cuda:{args.cuda_number}' if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    print(model)
    #torch.cuda.empty_cache()
    #model.to(device)
    
    print(f"\n {args.hggfc_model_name} Model Loaded from directory {args.loading_model_path}")    
    
    #------------------------------------------------------------------------------------------

    #DONT RUN THIS CELL UNTIL YOU WANT TO CREATE A NEW SET OF INPUTS (THIS MAY TAKE ~ 13 MINUTES)
    #from transformers import *
    #tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = True)
    model.eval()
    
    #random.seed(100)
    #for array in np.array_split(train_set, args.splits):
    """
    if args.get_train_data:
        trainData, train_labels, train_indexes = generate_training_data(train_data,
                                                                        model,
                                                                        device,
                                                                        tokenizer, 
                                                                        text_column = text_column,
                                                                        label_column=label_, 
                                                                        length = args.length, 
                                                                        overlap = args.overlap, 
                                                                        maxlen = args.maxlen,
                                                                        combine_layers = args.combine_layers,
                                                                        layers_from_end = args.layers_from_end)
        np.save(path_train_dat+f"train_{split_name}_.npy", trainData)
        np.save(path_train_dat+f"train_labels_{split_name}_.npy",train_labels)
        np.save(path_train_dat+f"train_indexes_{split_name}_.npy",train_indexes)
        del trainData, train_labels, train_indexes
    """
    if args.get_validation_data:
        validationData, validation_labels, validation_indexes = generate_training_data(validation_data,
                                                                                     model,
                                                                                     device,
                                                                                     tokenizer, 
                                                                                     text_column = text_column, 
                                                                                     label_column=label_,
                                                                                     remove_first = remove_first,
                                                                                     length = args.length, 
                                                                                     overlap = args.overlap, 
                                                                                     maxlen = args.maxlen,
                                                                                     combine_layers = args.combine_layers,
                                                                                     layers_from_end = args.layers_from_end)
        np.save(path_train_dat+"validation_.npy", validationData)
        np.save(path_train_dat+"validation_labels_.npy",validation_labels)
        np.save(path_train_dat+"validation_indexes_.npy",validation_indexes)
        del validationData, validation_labels, validation_indexes
    
    if args.get_test_data:
        testData, test_labels, test_indexes = generate_training_data(test_data,
                                                                     model,
                                                                     device,
                                                                     tokenizer, 
                                                                     text_column = text_column, 
                                                                     label_column = label_,
                                                                     remove_first = remove_first,
                                                                     length = args.length, 
                                                                     overlap = args.overlap, 
                                                                     maxlen = args.maxlen,
                                                                     combine_layers = args.combine_layers,
                                                                     layers_from_end = args.layers_from_end)


        np.save(path_train_dat+"test_.npy",testData)
        np.save(path_train_dat+"test_labels_.npy",test_labels)
        np.save(path_train_dat+"test_indexes_.npy",test_indexes)
        del testData, test_labels, test_indexes
        
    """ 
    if args.get_explanation_data:
        expData, exp_labels, exp_indexes = generate_training_data(exp_data,
                                                                     model,
                                                                     device,
                                                                     tokenizer, 
                                                                     text_column = text_column, 
                                                                     label_column = label_,
                                                                     length = args.length, 
                                                                     overlap = args.overlap, 
                                                                     maxlen = args.maxlen,
                                                                     combine_layers = args.combine_layers,
                                                                     layers_from_end = args.layers_from_end)


        np.save(path_train_dat+"exp_.npy",expData)
        np.save(path_train_dat+"exp_labels_.npy",exp_labels)
        np.save(path_train_dat+"exp_indexes_.npy",exp_indexes)
        del expData, exp_labels, exp_indexes
    """
    
    print("")
    print(f"\nExtraction for {args.dataset_subset} complete!")
    print("\nAll extraction complete!")
    
if __name__ == "__main__":
    random.seed(100)
    main()
