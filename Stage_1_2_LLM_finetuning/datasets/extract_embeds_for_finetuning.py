import os
import pandas as pd
from collections import Counter
from tqdm import tqdm
import pandas as pd
import datasets
import numpy as np
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer,LabelBinarizer
from ast import literal_eval 

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
def invert_multi_hot(encoded_labels,vocab):
    """Reverse a single multi-hot encoded label to a tuple of original labels."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)

def convert_to_binarised_labels(
    data,
    column: str,
    vocabulary: list = None,
    lookup = None,
    return_lookup: bool = False,
    problem_type: str = "multi_class"
): #or multi_label
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
def generate_attention_mask(input_ids,padding_token:int=0):
    attention_masks = []
    for id in input_ids:
        masks = [int(token_id != padding_token) for token_id in id]
        attention_masks.append(masks)
    return attention_masks
    
def generate_input_ids(
    tokens, 
    tokenizer, 
    length: int, 
    overlap: int, 
    maxlen: int, 
    indexing: bool = True,
    base_model:str = "default",
    incorporate_zero_padding = True
):
    tokenised_chunks = []
    left = 0
    right = length
    indexer = 0
    while(left<len(tokens)):
        tokenised_chunks.append(tokens[left:min(right,len(tokens))])
        left+=length-overlap #100 overlap
        right+=length-overlap
        indexer+=1
  
    if base_model in ["bert"]:
        CLS = tokenizer.cls_token
        SEP = tokenizer.sep_token
        pad_value = 0
    elif base_model in ["default","gpt"]:
        CLS = tokenizer.bos_token
        SEP = tokenizer.eos_token
        if incorporate_zero_padding:
            pad_value = 0
        else:
            pad_value = tokenizer.convert_tokens_to_ids(SEP)
        
    encoded_chunks = []
    for t in tokenised_chunks:
        if incorporate_zero_padding or (base_model in ["default","bert"]):
            t = [CLS] + t + [SEP]
        #else:
        #    t = t
        enc_chunk = tokenizer.convert_tokens_to_ids(t)
        encoded_chunks.append(enc_chunk)

    encoded_chunks = pad_sequences(encoded_chunks, maxlen=maxlen, value = pad_value, dtype='long', padding='post')
    attention_mask = generate_attention_mask(encoded_chunks, padding_token=pad_value) 
    if indexing:
        return encoded_chunks, attention_mask, indexer
    else:
        return encoded_chunks, attention_mask, None

def generate_training_data(
    df,
    tokenizer,
    text_column:str,
    label_column:str,
    remove_first: bool = False,
    length: int = 510,
    overlap: int = 100,
    maxlen = 512,
    indexing: bool = True,
    cumulative: bool = False,
    base_model: str = "default",
    incorporate_zero_padding: bool = True
):
    input_ids, attention_mask, labels, indexes = [], [], [], []
    for i in tqdm(range(len(df[text_column]))):
        texts = df[text_column].iloc[i]
        tokens = tokenizer.tokenize(texts)

        if remove_first:
            if(len(tokens) > 10000):
                tokens = tokens[len(tokens)-10000:]
        
        _input_ids, _attention_masks, index = generate_input_ids(
            tokens,
            tokenizer,
            length = length,
            overlap = overlap,
            maxlen = maxlen,
            indexing = indexing,
            base_model = base_model,
            incorporate_zero_padding = incorporate_zero_padding
        )
        #print("length mask:{}".format(len(_attention_masks)))
        #print("length input ids: {}".format(len(_input_ids)))
        document_label = df[label_column].iloc[i]
        if cumulative:
            input_ids.append(_input_ids)
            attention_mask.append(_attention_masks)
            labels.append(document_label)
            indexes.append(index)
        else:
            for i in range(len(_input_ids)):
                input_ids.append(_input_ids[i])
                attention_mask.append(_attention_masks[i])
                labels.append(document_label)
                indexes.append(index)
    if indexing:
        return input_ids, attention_mask, labels, indexes
    else:
        return input_ids, attention_mask, labels, None

def get_indexes(
    tokens,
    length: int,
    overlap: int
):
    left = 0
    right = length
    indexer = 0
    while(left<len(tokens)):
        left+=length-overlap #100 overlap
        right+=length-overlap
        indexer+=1
    return indexer

def create_input_ids(
    df,
    tokenizer,
    text_column:str,
    label_column:str,
    remove_first: bool = False,
    length: int = 510,
    overlap: int = 100,
    maxlen = 512,
    indexing: bool = True,
    testing_data: str = '',
    base_model:str = "default",
    incorporate_zero_padding = True
):
  
    input_ids, labels, indexes = [], [], []
    #lengths = []

    for i in tqdm(range(len(df[text_column]))):
        texts = df[text_column].iloc[i]
        tokens = tokenizer.tokenize(texts)
        index_ = get_indexes(tokens,length,overlap)
    
        if base_model in ["bert"]:
            CLS = tokenizer.cls_token
            SEP = tokenizer.sep_token
            pad_value = 0
        elif base_model in ["default","gpt"]:
            CLS = tokenizer.bos_token
            SEP = tokenizer.eos_token
            if incorporate_zero_padding:
                pad_value = 0
            else:
                pad_value = tokenizer.convert_tokens_to_ids(SEP)

        if(len(tokens) > length) and testing_data in ['last','']:
            tokens = tokens[len(tokens)-length:]
        elif(len(tokens) > length) and testing_data == 'first':
            tokens = tokens[:length]
        elif(len(tokens) > length) and testing_data == 'middle':
            start=max(1,int((len(tokens) - length)/2))
            end=min(start+length-1, len(tokens))
            tokens = tokens[start:end]
        
        if incorporate_zero_padding or (base_model in ["default","bert"]):
            tokens = [CLS] + tokens + [SEP]
        
        encoded_sent = tokenizer.convert_tokens_to_ids(tokens)
        input_ids.append(encoded_sent)
        #lengths.append(len(encoded_sent))
        document_label = df[label_column].iloc[i]
        labels.append(document_label)
        indexes.append(index_)
        
    input_ids = pad_sequences(
        input_ids,
        maxlen=maxlen,
        value=pad_value,
        dtype="long",
        padding="post"
        )#, truncating="pre"
  
    return input_ids, labels, indexes

#------------------------------------------Parameters-------------------------------------------------------------------------------------------------------------------------

def main(
    dataset_names: list,
    hggfc_model_name: str,
    save_dir: str,
    maxlen: int,
    n_c:int=0,
    testing_data = '',
    create_train_data: bool = False,
    create_test_data: bool = False,
    overlap = 100,
    #cumulative_test: bool = False
    data_source:str = None, #"Path to data.",
    incorporate_zero_padding:bool=True 
):

    """
    If testing_data == 'cumulative': 
        Prepare the tokenised data with chunks accumulated together for each document resembling the original document.
    if testing_data == 'last':
        Prepare the tokenised data with chunks from last part for each document.
    if testing_data == 'middle':
        Prepare the tokenised data with chunks from middle part for each document.
    if testing_data == 'first':
        Prepare the tokenised data with chunks from first part for each document.

    incorporate_zero_padding:bool=False 
        We used 0 post padding while training the gpt models. 
        If you do not want to incorporate this into your current replication you can choose to set this variable to 'True'. 
        Do keep in mind that the model (we released) were trained with this padding. 
        So when using those models, set this varible here to True/False and check for the performance on finetuning, whichever is similar to ours.

    """
    #for hggfc_model_nam in hggfc_model_name:
    print("Extraction started!\n")
    #hggfc_model_name = 'EleutherAI/gpt-j-6B'
    print(f"\nLoading {hggfc_model_name} model's tokenizer")
    model_path = "/gpfsdswork/dataset/HuggingFace_Models/"+hggfc_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #if tokenizer.pad_token is None:
    #    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.special_tokens_map)
    print("\nTokenizer Loaded")

    if hggfc_model_name in ["nlpaueb/legal-bert-base-uncased","law-ai/InLegalBERT","google-bert/bert-base-uncased"]:
        base_model = "bert"
    elif hggfc_model_name in ["EleutherAI/gpt-neo-1.3B","EleutherAI/gpt-neo-2.7B","EleutherAI/gpt-j-6B"]:
        base_model = "gpt"
    else:
        base_model = "default"

    #Choose strategy
    data_strategy = 0
    strat = data_strategy
    data_sub_strategy = 0
    #"if oversample or not after combining 'multi' and 'single' dataset"
    hggfc_model_name=hggfc_model_name.replace("/","_")

    
    maxlen = maxlen # the maximum length of the input to the model. 512 for BERT based, 2048 for GPT-Neo, GPT-J.
    length = maxlen-2 # the length of the input chunk = maxlen-2
    overlap = overlap # amount of overlap (in number of 'tokens') between adjacent chunks

    
    for dataset_subset in dataset_names:
        print(f"\nExtraction for {dataset_subset} started!")
        #dataset_subset = "scotus"
        #path_train_dat = save_dir+f"{dataset_subset}/{hggfc_model_name}/Strategy_{strat}/sub_strategy_{data_sub_strategy}/Training_data_{hggfc_model_name}/chunks_with_{overlap}_overlap_and_{maxlen}_input-length/"
        path_model_dat = save_dir+f"{dataset_subset}/{hggfc_model_name}/Strategy_{strat}/Training_data/chunks_with_{overlap}_overlap_and_{maxlen}_input-length_zero_pad/tuned_model_lr2e-06_warmup1000/"
        path_train_dat = save_dir+f"{dataset_subset}/{hggfc_model_name}/Strategy_{strat}/Training_data/chunks_with_{overlap}_overlap_and_{maxlen}_input-length_zero_pad/"
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #accelerator = Accelerator(project_dir=SAVE_DIR)       
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if os.path.isdir(path_model_dat)==False:
            os.makedirs(path_model_dat)
        #load dataset 
        if dataset_subset in ["eurlex","ecthr_a","ecthr_b"]:
            type_of_classification = "multi_label"#"multi_class"
            label_ = 'labels'
        elif dataset_subset in ["scotus"]:
            type_of_classification = "multi_class"
            label_ = 'labels'
        elif dataset_subset in ["ildc"]:
            type_of_classification = "binary"
            label_ = 'label'
        
        #load dataset 
        if dataset_subset in ["ecthr_a","ecthr_b","scotus"]:
            #dataset = datasets.load_from_disk(f"/gpfsdswork/dataset/HuggingFace/lex_glue/{dataset_subset}")
            dataset = datasets.load_dataset("lex_glue", dataset_subset)

            train_data = dataset['train'].to_pandas()
            validation_data = dataset['validation'].to_pandas()
            test_data = dataset['test'].to_pandas()
            text_column = train_data.columns[0]
            label_column = train_data.columns[1]
            """
            for doc in range(len(train_data)):
                labels = train_data['labels'][doc]
                for label in labels:
                    if label not in vocabulary:
                        vocabulary.append(label)
            vocabulary.sort()
            """

            vocabulary = get_vocabulary(
                train_data,
                label_column,
                problem_type = type_of_classification
            )

            train_data, lookup = convert_to_binarised_labels(
                train_data,
                label_column,
                vocabulary=vocabulary,
                return_lookup = True,
                problem_type = type_of_classification
            )

            validation_data = convert_to_binarised_labels(
                validation_data,
                label_column,
                vocabulary=vocabulary,
                lookup = lookup,
                problem_type = type_of_classification
            )#, lookup = lookup)

            test_data = convert_to_binarised_labels(
                test_data,
                label_column,
                vocabulary=vocabulary,
                lookup = lookup,
                problem_type = type_of_classification
            )

            #train_data = convert_to_binarised_labels(train_data, label_, return_lookup = False)
            #validation_data = convert_to_binarised_labels(validation_data, label_)#, lookup = lookup)
            #test_data = convert_to_binarised_labels(test_data, label_)#, lookup = lookup)
            if type_of_classification == "multi_label":
                train_data = concatenate_list_of_texts(train_data, column = "text")
                validation_data = concatenate_list_of_texts(validation_data, column = "text")
                test_data = concatenate_list_of_texts(test_data, column = "text")
            del dataset

            id2label = {idx:label for idx, label in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
            label2id = {label:idx for idx, label in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        if dataset_subset in ["ildc"]:
            if data_source is None:
                raise ValueError("Missing 'data_source'. Need path to ILDC data to use for extraction.")
            #Edit this if block
            data = pd.read_csv(data_source+"/ILDC_multi.csv")
            data_s = pd.read_csv(data_source+"/ILDC_single.csv")
            train_data = data.query("split=='train'")
            train_data_s = data_s.query("split=='train'")
            validation_data = data.query("split=='dev'")
            test_data = data.query("split=='test'")

            train_data = pd.concat([train_data, train_data_s], ignore_index=True)
            #train_data = train_data.append(train_data_s, ignore_index=True)
            del train_data_s, data_s, data
            label_column = label_
            text_column = 'text'
        
        print("\nDataset's slice view:\n")
        print(train_data)
        

        #DO NOT RUN THIS CELL UNTIL YOU WANT TO CREATE A NEW SET OF INPUTS (THIS MAY TAKE ~ 13 MINUTES)
        #from transformers import *
        #tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = True)

        if create_train_data:
            train_input_ids, train_attention_masks, train_labels, train_indexes = generate_training_data(
                train_data,
                tokenizer,
                text_column,
                label_column=label_,
                length = length,
                overlap = overlap,
                maxlen = maxlen,
                base_model = base_model
            )
            np.save(path_train_dat+"train_input_ids_.npy", train_input_ids)
            np.save(path_train_dat+"train_attention_masks_.npy", train_attention_masks)
            np.save(path_train_dat+"train_labels_.npy",train_labels)
            np.save(path_train_dat+"train_indexes_.npy",train_indexes)
            del train_input_ids, train_attention_masks, train_labels, train_indexes
            
        if create_test_data:
            if testing_data == "cumulative":
                validation_input_ids, validation_attention_masks, validation_labels, validation_indexes = generate_training_data(
                    validation_data, 
                    tokenizer, 
                    text_column, 
                    label_column=label_, 
                    length = length, 
                    overlap = overlap, 
                    maxlen = maxlen,
                    cumulative = True,
                    base_model = base_model,
                    incorporate_zero_padding = incorporate_zero_padding
                )
                np.save(path_train_dat+"validation_input_ids_"+testing_data+".npy",validation_input_ids)
                np.save(path_train_dat+"validation_attention_masks_"+testing_data+".npy",validation_attention_masks)
                np.save(path_train_dat+"validation_labels_"+testing_data+".npy",validation_labels)
                np.save(path_train_dat+"validation_indexes_"+testing_data+".npy",validation_indexes)
                del validation_input_ids, validation_attention_masks, validation_labels, validation_indexes
                
                
                test_input_ids, test_attention_masks, test_labels, test_indexes = generate_training_data(
                    test_data,
                    tokenizer,
                    text_column,
                    label_column=label_,
                    length = length,
                    overlap = overlap,
                    maxlen = maxlen,
                    cumulative = True,
                    base_model = base_model,
                    incorporate_zero_padding = incorporate_zero_padding
                )
                np.save(path_train_dat+"test_input_ids_"+testing_data+".npy",test_input_ids)
                np.save(path_train_dat+"test_attention_masks_"+testing_data+".npy",test_attention_masks)
                np.save(path_train_dat+"test_labels_"+testing_data+".npy",test_labels)
                np.save(path_train_dat+"test_indexes_"+testing_data+".npy",test_indexes)
                del test_input_ids, test_attention_masks, test_labels, test_indexes
            else:
                if incorporate_zero_padding or base_model in ["default","bert"]:
                    pad_value = 0
                else:
                    pad_value = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

                validation_input_ids, validation_labels, validation_indexes = create_input_ids(
                    validation_data,
                    tokenizer,
                    text_column,
                    label_column=label_,
                    length = length,
                    overlap = overlap,
                    maxlen = maxlen,
                    testing_data=testing_data,
                    base_model = base_model,
                    incorporate_zero_padding = incorporate_zero_padding
                )
                validation_attention_masks = generate_attention_mask(validation_input_ids, padding_token=pad_value) 
                np.save(path_train_dat+"validation_input_ids_"+testing_data+".npy",validation_input_ids)
                np.save(path_train_dat+"validation_attention_masks_"+testing_data+".npy",validation_attention_masks)
                np.save(path_train_dat+"validation_labels_"+testing_data+".npy",validation_labels)
                np.save(path_train_dat+"validation_indexes_"+testing_data+".npy",validation_indexes)
                del validation_input_ids, validation_attention_masks, validation_labels, validation_indexes
                
                
                test_input_ids, test_labels, test_indexes = create_input_ids(
                    test_data,
                    tokenizer,
                    text_column,
                    label_column=label_,
                    length = length,
                    overlap = overlap,
                    maxlen = maxlen,
                    testing_data=testing_data,
                    base_model = base_model,
                    incorporate_zero_padding = incorporate_zero_padding
                )
                test_attention_masks = generate_attention_mask(test_input_ids, padding_token=pad_value) 
                np.save(path_train_dat+"test_input_ids_"+testing_data+".npy",test_input_ids)
                np.save(path_train_dat+"test_attention_masks_"+testing_data+".npy",test_attention_masks)
                np.save(path_train_dat+"test_labels_"+testing_data+".npy",test_labels)
                np.save(path_train_dat+"test_indexes_"+testing_data+".npy",test_indexes)
                del test_input_ids, test_attention_masks, test_labels, test_indexes

        print("")
        print(f"\nExtraction for {dataset_subset} complete!")
    print("\nAll extraction complete!")
    
if __name__ == "__main__":
    testing_data_="cumulative"
    create_train_data = False
    create_test_data = True
    data_source = "MESc/experiments/datasets/ILDC"
    incorporate_zero_padding = True #Default True
    save_dir = 'MESc/experiments/finetuned_models/'
    main(
        dataset_names = ["ildc","ecthr_a","ecthr_b","scotus"], 
        hggfc_model_name = 'EleutherAI/gpt-neo-2.7B', 
        save_dir = save_dir, 
        maxlen = 2048, 
        testing_data=testing_data_,
        create_train_data = create_train_data,
        create_test_data = create_test_data,
        data_source=data_source,
        incorporate_zero_padding = incorporate_zero_padding
        )

    main(
        dataset_names = ["ildc","ecthr_a","ecthr_b","scotus"],
        hggfc_model_name = 'EleutherAI/gpt-neo-1.3B',
        save_dir = save_dir,
        maxlen = 2048,
        testing_data=testing_data_,
        create_train_data = create_train_data,
        create_test_data = create_test_data,
        data_source=data_source,
        incorporate_zero_padding = incorporate_zero_padding
        )

    main(
        dataset_names = ["ildc","ecthr_a","ecthr_b","scotus"],
        hggfc_model_name = 'EleutherAI/gpt-j-6B',
        save_dir = save_dir,
        maxlen = 2048,
        testing_data=testing_data_,
        create_train_data = create_train_data,
        create_test_data = create_test_data,
        data_source=data_source,
        incorporate_zero_padding = incorporate_zero_padding
        )

    main(
        dataset_names = ["ildc","ecthr_a","ecthr_b","scotus"],
        hggfc_model_name = 'EleutherAI/gpt-neo-2.7B',
        save_dir = save_dir,
        maxlen = 512,
        testing_data=testing_data_,
        create_train_data = create_train_data,
        create_test_data = create_test_data,
        data_source=data_source,
        incorporate_zero_padding = incorporate_zero_padding
        )

    main(
        dataset_names = ["ildc","ecthr_a","ecthr_b","scotus"],
        hggfc_model_name = 'EleutherAI/gpt-neo-1.3B',
        save_dir = save_dir,
        maxlen = 512,
        testing_data=testing_data_,
        create_train_data = create_train_data,
        create_test_data = create_test_data,
        data_source=data_source,
        incorporate_zero_padding = incorporate_zero_padding
        )

    main(
        dataset_names = ["ildc","ecthr_a","ecthr_b","scotus"],
        hggfc_model_name = 'EleutherAI/gpt-j-6B',
        save_dir = save_dir,
        maxlen = 512,
        testing_data=testing_data_,
        create_train_data = create_train_data,
        create_test_data = create_test_data,
        data_source=data_source,
        incorporate_zero_padding = incorporate_zero_padding
        )