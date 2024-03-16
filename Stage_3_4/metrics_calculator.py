##**Metrics Calculator**
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
import pandas as pd
# defining a function which calculates various metrics such as micro and macro precision, accuracy and f1
def metrics_calculator(preds, test_labels):
    cm = confusion_matrix(test_labels, preds)
    TP = []
    FP = []
    FN = []
    for i in range(0,2):
        summ = 0
        for j in range(0,2):
            if(i!=j):
                summ=summ+cm[i][j]

        FN.append(summ)
    for i in range(0,2):
        summ = 0
        for j in range(0,2):
            if(i!=j):
                summ=summ+cm[j][i]

        FP.append(summ)
    for i in range(0,2):
        TP.append(cm[i][i])
    precision = []
    recall = []
    for i in range(0,2):
        precision.append(TP[i]/(TP[i] + FP[i]))
        recall.append(TP[i]/(TP[i] + FN[i]))

    macro_precision = sum(precision)/2
    macro_recall = sum(recall)/2
    micro_precision = sum(TP)/(sum(TP) + sum(FP))
    micro_recall = sum(TP)/(sum(TP) + sum(FN))
    micro_f1 = (2*micro_precision*micro_recall)/(micro_precision + micro_recall)
    macro_f1 = (2*macro_precision*macro_recall)/(macro_precision + macro_recall)
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

import os

def absoluteFilePaths(directory):
  path_list = []
  for dirpath,_,filenames in os.walk(directory):
    for f in filenames:
      path_list.append(os.path.abspath(os.path.join(dirpath, f)))
  return path_list

def predict_and_plot(
    path_list,
    model,
    test_labels,
    data_generator,
    batches_per_epoch_test
): #, data_generator
    best = []
    max_f1 = 0

    """ Please use instead:* np.argmax(model.predict(x), axis=-1), 
        if your model does multi-class classification (e.g. if it uses a softmax last-layer activation).
        * (model.predict(x) > 0.5).astype("int32"), if your model does binary classification 
        (e.g. if it uses a sigmoid last-layer activation)."""
    print(f"test print: test label {len(test_labels)}")
    preds = model.predict(data_generator, steps = batches_per_epoch_test, verbose=0)
    y_pred = preds > 0.5
    print(f"test print: predicted label {len(preds)}")
    print(f"test print: predicted label > 0.5 {len(y_pred)}")

    print('  macro_precision,     macro_recall,        macro_f1,        micro_precision,      micro_recall,        micro_f1')
    macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(y_pred, test_labels)#[:-1])
    print(macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)
    matrix = classification_report(y_pred.flatten(), test_labels, digits=4)#[:-1])
    print('Classification Report\n:', matrix)


def predict_plot(
    model,
    test_labels,
    data_generator,
    batches_per_epoch_test,
    problem_type: list = ['multi-class']
): #num_label: int = None, 
    """ 
    Please use instead:
        * np.argmax(model.predict(x), axis=-1), 
        if your model does multi-class classification (e.g. if it uses a softmax last-layer activation).
        * (model.predict(x) > 0.5).astype("int32"), if your model does binary classification 
        (e.g. if it uses a sigmoid last-layer activation).
    """
    #if num_label!=None:
        #vocab_ = [item for item in range(num_label)]
    #else:
        #vocab_ = [item for item in range(len(test_labels[0]))]
    print(f"test print: test label {len(test_labels)}")
    preds = model.predict(data_generator, steps = batches_per_epoch_test, verbose=0)
    test_labels = test_labels.astype("int32")
    #print(preds)
    prediction = preds
    #if problem_type == 'multi-class':
    #    prediction = np.argmax(preds, axis=-1)
    #    test_labels = np.argmax(test_labels, axis=-1)
    #else:#if problem_type == 'multi-label':
    prediction = (preds > 0.5).astype("int32")
        #THRESHOLD = 0.5
        #prediction[preds > THRESHOLD] = 1
        #prediction[preds <= THRESHOLD] = 0
    #print(prediction)
    print('Classification Report\n:')
    if problem_type in ["multi_class", "multi_label"]:
        report = classification_report(test_labels, prediction, digits = 4, output_dict=True) #labels=vocab_, 
    elif problem_type in ["binary"]:
        report = classification_report(test_labels.flatten(), prediction, digits = 4, output_dict=True)
    #print(report)
    df_report = pd.DataFrame(report).transpose()
    #report = classification_report(test_labels, prediction, digits = 4, output_dict=True) #labels=vocab_,
    print(report)
    return df_report, prediction 

"""
def pt_model_predict(model, dataloader, ):
    model.eval()
    predictions , true_labels = [], []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():        
          outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.logits#[0]#.sigmoid()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.float().to('cpu').numpy()
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    print('    DONE.')

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    prediction = predictions
    THRESHOLD = 0.5
    prediction[predictions > THRESHOLD] = 1
    prediction[predictions <= THRESHOLD] = 0
    res = multi_label_metrics(predictions, true_labels)
    print(res)
    result = multilabel_confusion_matrix(true_labels, predictions)

    print(classification_report(true_labels, prediction, labels=vocabulary, digits = 4))
    
    return predictions, true_labels
"""