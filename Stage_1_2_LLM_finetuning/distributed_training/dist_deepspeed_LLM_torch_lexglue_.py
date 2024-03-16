#For running in multiple gpus
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import extract_model_from_parallel

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


import argparse
import os
import pandas as pd
import time
import logging
from tqdm import tqdm
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences

import datasets

import numpy as np
from transformers import AutoTokenizer 
from transformers import AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer,LabelBinarizer

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from ast import literal_eval

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune step: LLM (GPT-Neo, GPT-J) on a sequence classification task")
    
    parser.add_argument(
        "--to_train",
        type=bool,
        default=False,
        help="to train the LLM?"
    )
    parser.add_argument(
        "--freeze_all",
        type=bool,
        default = False,
        help = "If to freeze all layers of gpt neo/J"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch size to train the LLM?"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="learning rate to train the LLM?"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="epochs to train the LLM?"
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=1000,
        help="number of warmup steps to train the LLM?"
    )
    parser.add_argument(
        "--to_test",
        type=bool,
        default=False,
        help="to test the LLM?"
    )
    parser.add_argument(
        "--test_input_len",
        type=int,
        default=512,
        help="input length to test the LLM?"
    )
    parser.add_argument(
        "--testing_model_path",
        type=str,
        default=None,
        help="path to the finetuned LLM to test"
    )
    parser.add_argument(
        "--testing_model_epoch",
        type=int,
        default=None,
        help="epoch of the finetuned LLM to test"
    )
    parser.add_argument(
        "--testing_data_",
        type=str,
        default=None,
        help="typr of data to test"
    )
    parser.add_argument(
        "--load_and_retrain",
        type=bool,
        default=False,
        help="To continue training from a previous checkpoint?"
    )
    parser.add_argument(
        "--retraining_model_path",
        type=str,
        default=None,
        help="path to the LLM model to load and retrain"
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
        "--data_path", 
        type=str, 
        default=None, 
        help="path to the training data's directory"
    )
    parser.add_argument(
        "--minimum_number_of_chunks", 
        type=int, 
        default=0, 
        help="documents with minimum number of 'n' chunk to consider for testing. While only the last chunk will be used for testing but this parameter is used to test the LLM only on documents having minumum number of 'n' chunks"
    )
    parser.add_argument(
        "--chunk_number", 
        type=int, 
        default=None, 
        help="documents with minimum number of 'minimum_number_of_chunks' chunk to consider for testing. While only the 'chunk_number' of a document will be used for testing."
    )
    parser.add_argument(
        "--start_chunk", 
        type=int, 
        default=None, 
        help="Starting chunk to consider for testing. While only the 'chunk_number' of a document will be used for testing."
    )
    parser.add_argument(
        "--hggfc_model_name",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--SAVE_DIR",
        type=str,
        default=None,
        help="Path to save the model and evaluation results in"
    )
    parser.add_argument(
        "--trained_with_deepspeed_accelerate",
        type=bool,
        default=None,
        help="Trained with deepspeed in accelerate"
    )
    parser.add_argument(
        "--trained_with_accelerate",
        type=bool,
        default=None,
        help="trained only with accelerate"
    )
    parser.add_argument(
        "--trained_without_accelerate",
        type=bool,
        default=None,
        help="trained without distributed computing (accelerate)"
    )
    parser.add_argument(
        "--convert_to_torch_model",
        type=bool,
        default=False,
        help="save the distributed model to a 32 bit torch model"
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_subset is None and args.data_path:
        raise ValueError("Need either a dataset name and a training/validation file (both should match (please check in the data_path string for the dataset_name to be matching)).")
        
    if (args.to_train or args.to_test) and args.SAVE_DIR is None:
        if args.freeze_all:
            args.SAVE_DIR = args.data_path+f"freezed_model_lr{args.learning_rate}_warmup{args.num_warmup_steps}/"
        else:
            args.SAVE_DIR = args.data_path+f"tuned_model_lr{args.learning_rate}_warmup{args.num_warmup_steps}/"
        if os.path.isdir(args.SAVE_DIR)==False:
            os.makedirs(args.SAVE_DIR)
    
    if args.testing_data_ is None:
        args.testing_data_ = ''
    #if args.to_train==False and args.to_test==False:
    #    raise ValueError("to_train and to_test are both False")
        
    if args.load_and_retrain and args.retraining_model_path==False:
        raise ValueError("The path to load the checkpoint to continue trining is not given. Give --retraining_model_path")
    print("-"*20)
    print("Report")
    print("-"*20)
    print(args)
    print("-"*20)
    #if args.to_train==False and args.to_test==True:
    return args


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def checkpoint_model(
    checkpoint_folder,
    ckpt_id,
    model,
    epoch,
    last_global_step = 0,
    **kwargs
    ):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return

def convert_to_binarised_pred(logits):
    binarised = np.zeros_like(logits)
    am = np.argmax(logits, axis=-1)
    #print(am)
    for i, (am_, logit) in enumerate(zip(am, logits)):
        binarised[i][am_] = 1
    return binarised
    
def evaluate(
    model, 
    eval_dataloader, 
    accelerator, 
    per_device_eval_batch_size, 
    save_path, 
    epoch: int = None, 
    name: str = None, 
    labels=None, 
    problem_type: str = "multi_label", 
    test_input_len: int = 512, 
    minimum_number_of_chunks: int = 0,
    chunk_number: int = None,
    testing_data_: str = ''
    ):

    device = accelerator.device
    print("\n")
    print("----running evaluation----")
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    #losses = []
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch) #.to(device)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        #loss = outputs.loss
            """
            The current method is using numpy.dtype('float32') and then converting it to uint16 values for ONNX, because native NumPy 
            does not support bfloat16 yet: numpy/numpy#19808.
            """
        #print(f"logits:{outputs.logits}")
        if problem_type == "multi_label":
            logits = torch.sigmoid(outputs.logits).double()#.argmax(dim=-1)
        elif problem_type == "multi_class":
            logits = torch.softmax(outputs.logits, dim=-1).double()
        elif problem_type == "binary":
            logits = outputs.logits.argmax(dim=-1)
            #logits = np.asarray(logits)
        logits = logits.detach().cpu().numpy()
        #logits = torch.tensor(logits.detach().cpu())#.numpy()
        #predictions.append(accelerator.gather_for_metrics(logits.repeat(per_device_eval_batch_size)))
        #losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)))
        if problem_type == "multi_class":
            logits = convert_to_binarised_pred(logits)
        predictions.append(logits)
        label_ids = b_labels.float().to('cpu').numpy()
        #label_ids = torch.tensor(b_labels.float().to('cpu'))
        true_labels.append(label_ids)
        #print("------------\n")
        #print(f"predictions:{predictions}")
        #print(f"predictions size:{predictions.size}")
        #print(f"true_labels:{true_labels}")
        #print(f"true_labels size:{true_labels.size}")
        #print("------------\n")
        

    #losses = torch.cat(losses)
    #predictions = torch.cat(predictions)
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)#torch.cat(true_labels)
    #predictions = torch.cat(predictions, axis=0)
    #true_labels = torch.cat(true_labels, axis=0)

    # Report the final accuracy for this validation run.
    if problem_type in ["multi_label", "binary"]:
        THRESHOLD = 0.5
        predictions[predictions > THRESHOLD] = 1
        predictions[predictions <= THRESHOLD] = 0
    #print("---------------------------------FULL PREDICTIONS-------------------------------\n")
    #print(f"predictions:{predictions.shape}")
    #print(f"predictions size:{predictions.size}")
    #print(f"true_labels:{true_labels.shape}")
    #print(f"true_labels size:{true_labels.size}")
    #print("--------------------------------------------------------------------------------\n")
    """
    output_dir = f"ZeRO3_epoch_{epoch}_predictions/"
    output_dir = save_path + output_dir
    #output_dir = os.path.join(save_path, output_dir)
    if os.path.isdir(output_dir)==False:
        os.makedirs(output_dir)
    """
    if minimum_number_of_chunks>0:
        testing_data_ = ''
    if chunk_number == None:
        chunk_number = "last"

    if epoch!=None:
        np.save(save_path+f"{name}_predictions_{epoch}__final_{test_input_len}_input_len_{minimum_number_of_chunks}_chunksorMore_onChunk_{chunk_number}_{testing_data_}.npy", predictions)
        np.save(save_path+f"{name}_true_labels_{epoch}__final_{test_input_len}_input_len_{minimum_number_of_chunks}_chunksorMore_onChunk_{chunk_number}_{testing_data_}.npy", true_labels)
    else:
        np.save(save_path+f"{name}_predictions__loaded_{test_input_len}_input_len_{minimum_number_of_chunks}_chunksorMore_{testing_data_}.npy", predictions)
        np.save(save_path+f"{name}_true_labels__loaded_{test_input_len}_input_len_{minimum_number_of_chunks}_chunksorMore_{testing_data_}.npy", true_labels)
    print("\n")
    if problem_type in ["multi_class", "multi_label"]:
        report = classification_report(true_labels, predictions, labels=labels, digits = 4, output_dict=True)
    elif problem_type in ["binary"]:
        report = classification_report(true_labels.flatten(), predictions, digits = 4, output_dict=True)
    #print(report)
    df_report = pd.DataFrame(report).transpose()
    print(df_report)
    if epoch!=None:
        df_report.to_csv(save_path+f"{name}_report_{epoch}__final_{test_input_len}_input_len_{minimum_number_of_chunks}_chunksorMore_onChunk_{chunk_number}_{testing_data_}.csv", index= True)
    else:
        df_report.to_csv(save_path+f"{name}_report__loaded_{test_input_len}_input_len_{minimum_number_of_chunks}_chunksorMore_onChunk_{chunk_number}_{testing_data_}.csv", index= True)
    print("----Done----\n")


def get_vocabulary(
    df,
    column,
    problem_type: str = 'multi_label'
    ):
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

def get_only_large_documents(
    number_of_chunks,
    indexes,
    input_ids,
    attention_masks,
    labels,
    chunk_number:int=None
    ):
    """
    chunk_number = {1, ... the last chunk}.
    So for slicing chunk_number-1 will be used.
    """
    if chunk_number == None:
        # Take the last chunk
        chunk_number = number_of_chunks-1
    indexes_ , input_ids_, attention_masks_, labels_ = [],[],[],[]
    for i, (index, input_id, att_mask, label) in enumerate(zip(indexes, input_ids, attention_masks, labels)):
        if index >= number_of_chunks:
            indexes_.append(index)
            input_ids_.append(input_id[chunk_number-1]) 
            attention_masks_.append(att_mask[chunk_number-1])
            labels_.append(label)
    #indexes_ = np.asarray(indexes_)#, dtype=object)
    #input_ids_ = np.asarray(input_ids_)#, dtype=object)
    #attention_masks_ = np.asarray(attention_masks_)#, dtype=object)
    #labels_ = np.asarray(labels_)#, dtype=object)
    return indexes_ , input_ids_, attention_masks_, labels_

def main_(args):#strat:int, data_path:str, dataset_subset: str ,hggfc_model_name: str): 
    
    checkpoint_path = args.retraining_model_path
    model_path = args.hggfc_model_name

    if args.dataset_subset in ["eurlex","ecthr_a","ecthr_b"]:
        type_of_classification = "multi_label"#"multi_class"
        label_ = 'labels'
    elif args.dataset_subset in ["scotus"]:
        type_of_classification = "multi_class"
        label_ = 'label'
    elif args.dataset_subset in ["ildc"]:
        type_of_classification = "binary"
        
    
    #load dataset
    vocabulary = []
    if args.dataset_subset in ["eurlex","ecthr_a","ecthr_b","scotus"]:
        #dataset = datasets.load_from_disk(f"/gpfsdswork/dataset/HuggingFace/lex_glue/{args.dataset_subset}")
        dataset = datasets.load_dataset("lex_glue", args.dataset_subset)
        train_data = dataset['train'].to_pandas()
        label_column = train_data.columns[1]

        vocabulary = get_vocabulary(train_data, label_column, problem_type = type_of_classification)
        print(train_data)
        del train_data, dataset 
    
    accelerator = Accelerator(project_dir=args.SAVE_DIR)#,automatic_checkpoint_naming=True)

    #load data
    #loading the stored chunks
    if args.to_train:
        train_inputs = np.load(args.data_path+'train_input_ids_.npy',allow_pickle=True)
        train_masks = np.load(args.data_path+'train_attention_masks_.npy',allow_pickle=True)
        train_labels = np.load(args.data_path+'train_labels_.npy',allow_pickle=True)
        train_indexes = np.load(args.data_path+'train_indexes_.npy',allow_pickle=True)
        #print(f"train_labels:{train_labels}")
    if args.to_test:
        test_inputs = np.load(args.data_path+'test_input_ids_'+args.testing_data_+'.npy',allow_pickle=True)
        test_masks = np.load(args.data_path+'test_attention_masks_'+args.testing_data_+'.npy',allow_pickle=True)
        test_labels = np.load(args.data_path+'test_labels_'+args.testing_data_+'.npy',allow_pickle=True)
        test_indexes = np.load(args.data_path+'test_indexes_'+args.testing_data_+'.npy',allow_pickle=True)
    validation_inputs = np.load(args.data_path+'validation_input_ids_'+args.testing_data_+'.npy',allow_pickle=True)
    validation_masks = np.load(args.data_path+'validation_attention_masks_'+args.testing_data_+'.npy',allow_pickle=True)
    validation_labels = np.load(args.data_path+'validation_labels_'+args.testing_data_+'.npy',allow_pickle=True)
    validation_indexes = np.load(args.data_path+'validation_indexes_'+args.testing_data_+'.npy',allow_pickle=True)
        
    if args.to_train:
        train_inputs = np.asarray(train_inputs)
        train_masks = np.asarray(train_masks)
        
        train_inputs = torch.tensor(train_inputs)#.type(torch.LongTensor)
        train_labels = torch.tensor(train_labels, dtype=torch.int64)#.type(torch.LongTensor) #int64
        train_masks = torch.tensor(train_masks)#.type(torch.LongTensor)
        
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = args.batch_size)
        print(f"train_labels:{train_labels.shape}")
        del train_inputs, train_masks, train_labels, train_data, train_sampler
    
    if args.minimum_number_of_chunks > 0:
        if args.to_test:
            test_indexes, test_inputs, test_masks, test_labels = get_only_large_documents(
                args.minimum_number_of_chunks,
                test_indexes,
                test_inputs,
                test_masks,
                test_labels,
                chunk_number=args.chunk_number)
        validation_indexes, validation_inputs, validation_masks, validation_labels = get_only_large_documents(
            args.minimum_number_of_chunks,
            validation_indexes,
            validation_inputs,
            validation_masks,
            validation_labels,
            chunk_number=args.chunk_number)

    if args.to_test:
        test_inputs = np.asarray(test_inputs)
        test_masks = np.asarray(test_masks)
        
        test_inputs = torch.tensor(test_inputs)#.type(torch.LongTensor)
        test_labels = torch.tensor(test_labels, dtype=torch.int64)#.type(torch.LongTensor) #int64
        test_masks = torch.tensor(test_masks)#.type(torch.LongTensor)
        
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size = args.batch_size)
        del test_inputs, test_masks, test_labels, test_data, test_sampler 
        
    validation_inputs = np.asarray(validation_inputs)
    validation_masks = np.asarray(validation_masks)
    
    validation_inputs = torch.tensor(validation_inputs)#.type(torch.LongTensor)
    validation_labels = torch.tensor(validation_labels)#.type(torch.LongTensor)
    validation_masks = torch.tensor(validation_masks)#.type(torch.LongTensor)
    
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = args.batch_size)
   
    del validation_inputs, validation_masks, validation_labels
    del validation_data, validation_sampler
    
    #load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if type_of_classification == "multi_label":
        if args.dataset_subset in ["ecthr_a","ecthr_b"]:
            id2label = {idx:label for idx, label in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
            label2id = {label:idx for idx, label in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            problem_type="multi_label_classification", 
            num_labels=len(vocabulary),
            id2label=id2label,
            label2id=label2id,
            #ignore_mismatched_sizes = True
            )
    #for multi-class classification
    elif type_of_classification == "multi_class":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            num_labels=len(vocabulary),
            #ignore_mismatched_sizes = True
            )
    elif type_of_classification == "binary":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            #ignore_mismatched_sizes = True
            )
    
    if tokenizer.pad_token is None:
        # set the pad token of the model's configuration
        model.config.pad_token_id = model.config.eos_token_id
    del tokenizer
    
    #preparing for distributed training
    #model = accelerator.prepare(model)
    if args.to_train:
        if args.freeze_all:
            for param in model.transformer.parameters():
                param.requires_grad = False
        #training parameters
        #lr = args.learning_rate   
        max_grad_norm = 1.0
        #epochs = args.epochs
        num_total_steps = len(train_dataloader)*args.epochs
        #num_warmup_steps = args.num_warmup_steps
        #warmup_proportion = float(args.num_warmup_steps) / float(num_total_steps)  # 0.1
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.num_warmup_steps, num_training_steps = num_total_steps)

        #not optimised for accelerate 
        if args.load_and_retrain:
            checkpoint = torch.load(args.retraining_model_path)
            continue_from_epoch = checkpoint['epoch']
            accelerator.load_state(args.SAVE_DIR)
            # Assume the checkpoint was saved 100 steps into the epoch
            #accelerator.skip_first_batches(train_dataloader, 100)

            #model.load_state_dict(checkpoint['model'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            #scheduler = checkpoint['lr_scheduler']
            del checkpoint
            torch.cuda.empty_cache()
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #model.to(device)
        else:
            continue_from_epoch = 1


        #---------------------------------------------------------------------------------------------------------------------------------------
        #for distributed training
        device = accelerator.device
        #model.to(device)
        model, train_dataloader, optimizer, scheduler = accelerator.prepare(
            model, train_dataloader, optimizer, scheduler)
        #---------------------------------------------------------------------------------------------------------------------------------------
        seed_val = 21

        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        #start training-----------------
        loss_values = []
        # For each epoch...
        for epoch_i in tqdm(range(0, args.epochs)):
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
            print('Training...')

            t0 = time.time()
            total_loss = 0

            model.train()

            for step, batch in enumerate(tqdm(train_dataloader)):
                #if step % 40 == 0 and not step == 0:
                    #print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))


                b_input_ids = batch[0]#.long()#.to(device)
                b_input_mask = batch[1]#.long()#.to(device)
                if type_of_classification == "binary":
                    b_labels = batch[2].long()
                else:
                    b_labels = batch[2].bfloat16()#.long()#.to(device)
                #print(b_input_ids.type())
                #print("\n--------------\n")
                #print(b_input_mask.type())
                #print("\n--------------\n")
                #print(b_labels.type())        

                #model.zero_grad()        
                #print(f"b_labels: {b_labels}")
                outputs = model(b_input_ids, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

                loss = outputs[0]
                total_loss += loss.item()
                #loss.backward()
                accelerator.backward(loss)
                #accelerator.clip_grad_norm_(model.parameters(), 1.0)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                get_accelerator().empty_cache()
            avg_train_loss = total_loss / len(train_dataloader)            
            loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.4f}".format(avg_train_loss))

            t0 = time.time()

            #----------------------------------------------------------------------------------------------------------------------
            #model_file_name = f'dist_torch-model_epoch-{epoch_i+continue_from_epoch}_.pth'
            #model_file_name = f'torch-model_epoch-{epoch_i+continue_from_epoch}_lr_{lr}.pth' #'torch-model_epoch1-epoch-3-epoch-{}.pth'.format(epoch_i+1) 
            print(args.SAVE_DIR)
            #if load_and_retrain:
            #    print(f"continue_from_epoch = {continue_from_epoch}")
            #    print(f"path_to_continue_ft: {args.retraining_model_path}") 

            #print(model_file_name)
            print(f"saving checkpoint for epoch: {epoch_i+1}")
            #path = os.path.join(SAVE_DIR, model_file_name)
            #torch.save(model.cpu().state_dict(), path) # saving model

            # Register the LR scheduler
            #accelerator.register_for_checkpointing(scheduler)
            output_dir = f"ZeRO3_epoch_{epoch_i}__final"
            output_dir = os.path.join(args.SAVE_DIR, output_dir)
            #accelerator.save_state(output_dir)
            
            if args.freeze_all:
                if args.dataset_subset in ["ildc"]:
                    # Save the DeepSpeed checkpoint to the specified path
                    checkpoint_model(output_dir, epoch_i+1, model, epoch_i+1)#, completed_steps)
            else:
                # Save the DeepSpeed checkpoint to the specified path
                checkpoint_model(output_dir, epoch_i+1, model, epoch_i+1)#, completed_steps)
            #----------------------------------------------------------------------------------------------------------------------------------- 
            #pred_dir = f"ZeRO3_epoch_{epoch_i}_predictions/"
            #pred_dir = SAVE_DIR + pred_dir
            #output_dir = os.path.join(save_path, output_dir)
            #if os.path.isdir(pred_dir)==False:
            #    os.makedirs(pred_dir)
            model.eval()
            print("\nRunning Validation...")
            evaluate(model, validation_dataloader, accelerator, per_device_eval_batch_size=args.batch_size, save_path = args.SAVE_DIR, epoch = epoch_i, name = 'val', labels = vocabulary, problem_type = type_of_classification, test_input_len = args.test_input_len, minimum_number_of_chunks = args.minimum_number_of_chunks)

            if args.to_test:
                print("\nRunning Test...")
                evaluate(model, test_dataloader, accelerator, per_device_eval_batch_size=args.batch_size, save_path = args.SAVE_DIR, epoch = epoch_i, name = 'test', labels = vocabulary, problem_type = type_of_classification, test_input_len = args.test_input_len, minimum_number_of_chunks = args.minimum_number_of_chunks)

        print("")
        print("Training complete!")

    if args.to_test==True and args.to_train==False:
        if args.testing_model_path != None:
            print("evaluating on loaded fine-tuned model")
            if args.trained_without_accelerate:
                checkpoint = torch.load(args.testing_model_path)['model']
                model.load_state_dict(checkpoint)
                del checkpoint
                model = accelerator.prepare(model)

            elif args.trained_with_accelerate:
                model.load_state_dict(torch.load(args.testing_model_path))
                
                model = accelerator.prepare(model)
                #old working without checkpoint .pth file
                #accelerator.load_state(args.testing_model_path)
            
            elif args.trained_with_deepspeed_accelerate:
                #dummy_dataloader = validation_dataloader
                #model, dummy_dataloader = accelerator.prepare(model, dummy_dataloader)
                #model = accelerator.unwrap_model(model)
                #model = load_state_dict_from_zero_checkpoint(model, args.testing_model_path)
                
                model.load_state_dict(torch.load(args.testing_model_path))
                
                #old working without checkpoint .pth file
                #model.load_state_dict(get_fp32_state_dict_from_zero_checkpoint(args.testing_model_path))
                model = accelerator.prepare(model)
                
            val_name = "with_ft_val"
            test_name = "with_ft_test"
            
        else:
            print("evaluating on initial pretrained model")
            model = accelerator.prepare(model)
            val_name = "without_ft_val"
            test_name = "without_ft_test"
            
        if args.convert_to_torch_model:
            print("converting to fp32 model")
            #please use full fp32 model loading instead of torch_dtype=torch_dtype=torch.float16 to save in full 32fp model
            model_save_path_ = os.path.join(args.testing_model_path, f"torch_model_with_32fp_wrapper.pth")
            torch.save(extract_model_from_parallel(model).cpu().state_dict(), model_save_path_)
            print("model saved")
            model = accelerator.prepare(model)

        model.eval()
        
        print(f"\nRunning Validation : {epoch_i+1} ...")
        evaluate(
            model, 
            validation_dataloader, 
            accelerator, 
            per_device_eval_batch_size=args.batch_size, 
            save_path = args.SAVE_DIR, 
            epoch = args.testing_model_epoch, 
            name = val_name, 
            labels = vocabulary, 
            problem_type = type_of_classification, 
            test_input_len = args.test_input_len, 
            minimum_number_of_chunks = args.minimum_number_of_chunks,
            testing_data_ = args.testing_data_,
            chunk_number = args.chunk_number
            )

        print(f"\nRunning Test for epoch : {epoch_i+1} ...")
        evaluate(
            model, 
            test_dataloader, 
            accelerator, 
            per_device_eval_batch_size=args.batch_size, 
            save_path = args.SAVE_DIR, 
            epoch = args.testing_model_epoch, 
            name = test_name, 
            labels = vocabulary, 
            problem_type = type_of_classification, 
            test_input_len = args.test_input_len, 
            minimum_number_of_chunks = args.minimum_number_of_chunks,
            testing_data_ = args.testing_data_,
            chunk_number = args.chunk_number
            )
        del model
        print("")
        print("Evaulation complete!")

def main():
    args = parse_args()
    #dict_args = vars(args)
    #dict_args = pd.DataFrame(dict_args).transpose()
    #dict_args.to_csv(args.SAVE_DIR+f"model_args.csv", index= True)
    """
    parameters:-
    
    strat: 1 (for 512 input length), and 0 (for 2048 input length)
    data_path: path to the training data's directory
    
    If testing_data == 'cumulative': 
        Test the tokenised data with chunks accumulated together for each document resembling the original document.
        Use 'minimum_number_of_chunks' and  'chunk_number' to decide which chunk of the document to use for testing.
    if testing_data == 'last':
        Test the tokenised data with chunks from last part for each document.
    if testing_data == 'middle':
        Test the tokenised data with chunks from middle part for each document.
    if testing_data == 'first':
        Test the tokenised data with chunks from first part for each document.
    """
    start_chunk = args.start_chunk if args.start_chunk is not None else 0
    if args.minimum_number_of_chunks>0 and args.chunk_number is None:
        for chunk_number in range(start_chunk, args.minimum_number_of_chunks):
            #if chunk_number in [3,4,6,8,10]:
            args.chunk_number = chunk_number
            main_(args)
    else:
        main_(args)
        
        
        
if __name__ == "__main__":
    main()