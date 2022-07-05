import numpy as np
import pandas as pd
import json
import os
import more_itertools as mit
try:
    import cPickle as pkl
except:
    import pickle as pkl


# https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


# Convert dataset into ERASER format: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/utils.py
def get_evidence(post_id, anno_text, explanations):
    output = []

    indexes = sorted([i for i, each in enumerate(explanations) if each==1])
    span_list = list(find_ranges(indexes))

    for each in span_list:
        if type(each)== int:
            start = each
            end = each+1
        elif len(each) == 2:
            start = each[0]
            end = each[1]+1
        else:
            print('error')

        output.append({"docid":post_id,
              "end_sentence": -1,
              "end_token": end,
              "start_sentence": -1,
              "start_token": start,
              "text": ' '.join([str(x) for x in anno_text[start:end]])})
    return output


# To use the metrices defined in ERASER, we will have to convert the dataset
def convert_to_eraser_format(dataset, method, save_split, save_path, id_division):
    """
    Args:
        dataset: is a list with:
            [docid,
             str_label,
             [101, # encoded tokens
              5310,
              25805,
              5582,
              4319,
              2224,
              2025,
              11382,
              3489,
              2012,
              2035,
              4283,
              2005,
              2008,
              19380,
              102],
             [[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # rationales
              [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # rationales
              [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]], # rationales
             ['str_label0', 'str_label1', 'str_label2']]
        method: to aggregate rationales -> str: "union"
        save_path: path where the dataset in Eraser Format will be stored
        save_split: boolean
        id_division: og file doc_ids fro the respective train/val/test spits
    """
    final_output = []

    if save_split:
        train_fp = open(save_path + 'train.jsonl', 'w')
        val_fp = open(save_path + 'val.jsonl', 'w')
        test_fp = open(save_path + 'test.jsonl', 'w')

    for tcount, eachrow in enumerate(dataset):
        temp = {}
        post_id = eachrow[0]
        post_class = eachrow[1]
        anno_text_list = eachrow[2]
        majority_label = eachrow[1]

        #if majority_label == 'normal':
        #    continue

        all_labels = eachrow[4]
        explanations = []
        for each_explain in eachrow[3]:
            explanations.append(list(each_explain))

        # For this work, we have considered the union of explanations. Other options could be explored as well.
        if method == 'union':
            final_explanation = [any(each) for each in zip(*explanations)]
            final_explanation = [int(each) for each in final_explanation]

        temp['annotation_id'] = post_id
        temp['classification'] = post_class
        temp['evidences'] = [get_evidence(post_id, list(anno_text_list), final_explanation)]
        temp['query'] = "What is the class?"
        temp['query_type'] = None
        final_output.append(temp)

        if save_split:
            if not os.path.exists(save_path + 'docs'):
                os.makedirs(save_path + 'docs')

            with open(save_path + 'docs/' + post_id, 'w') as fp:
                fp.write(' '.join([str(x) for x in list(anno_text_list)]))

            if post_id in id_division['train']:
                train_fp.write(json.dumps(temp) + '\n')

            elif post_id in id_division['val']:
                val_fp.write(json.dumps(temp) + '\n')

            elif post_id in id_division['test']:
                test_fp.write(json.dumps(temp) + '\n')
            else:
                print(post_id)

    if save_split:
        train_fp.close()
        val_fp.close()
        test_fp.close()
    print("Done")
    return final_output


def output_eraser_format(model):
    # Read test set
    x_test = np.load(os.path.join(data_path, "x_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))
    pred_test = np.load(os.path.join(data_path, "pred_test.npy"))
    rationales_test = np.load(os.path.join(data_path, "rationales_test.npy"))
    pred_rationales_test = np.load(os.path.join(data_path, f"x_test-{model}.npy"))
    pred_soft_rationales_test = np.load(os.path.join(data_path, f"x_test_soft-{model}.npy"))
    comprehensiveness_scores = np.load(os.path.join(data_path, f"comprehensiveness_test-{model}.npy"))
    sufficiency_scores = np.load(os.path.join(data_path, f"sufficiency_test-{model}.npy"))
    test_files = idx_to_file['test']
    code_to_class = {0: "normal", 1: 'offensive', 2: 'hatespeech'}
    dataset = []
    dataset_instance = {}
    for idx in test_files:
        dataset_instance = {}
        dataset_instance["annotation_id"] = test_files[idx] # not really sure what it means. Use document_id
        rationales_instance = {}
        rationales_instance['docid'] = test_files[idx] # document_id same as annotation id.
        inst_rationales = pred_rationales_test[idx][np.where(x_test[idx] == 1)[0][0] + 1:].astype(int)
        inst_soft_rationales = pred_soft_rationales_test[idx][np.where(x_test[idx] == 1)[0][0] + 1:]
        if np.argmax(pred_test[idx]) != 0:
            rationales_instance["hard_rationale_predictions"] = get_hard_rationale_predictions(inst_rationales)
            rationales_instance["soft_rationale_predictions"] = inst_soft_rationales.tolist()
        else:
            rationales_instance["hard_rationale_predictions"] = [{"start_token":-1, "end_token":-1}]
            rationales_instance["soft_rationale_predictions"] = np.zeros(inst_soft_rationales.shape).tolist()

        dataset_instance["rationales"] = [rationales_instance]
        dataset_instance["classification"] = code_to_class[np.argmax(pred_test[idx])]
        dataset_instance["classification_scores"] = {"normal": pred_test[idx][0],
                                                     "offensive": pred_test[idx][1],
                                                     "hatespeech": pred_test[idx][2]}
        dataset_instance["comprehensiveness_classification_scores"] = {"normal": comprehensiveness_scores[idx][0],
                                                                       "offensive": comprehensiveness_scores[idx][1],
                                                                       "hatespeech": comprehensiveness_scores[idx][2]}
        dataset_instance["sufficiency_classification_scores"] = {"normal": sufficiency_scores[idx][0],
                                                                 "offensive": sufficiency_scores[idx][1],
                                                                 "hatespeech": sufficiency_scores[idx][2]}
        dataset.append(dataset_instance)
    with open(f'data/output_{model}.jsonl', 'w') as outfile:
        for entry in dataset:
            json.dump(entry, outfile)
            outfile.write('\n')
    return print("Done")


def get_hard_rationale_predictions(array_rationale):
    list_rationales = []
    fragment = {}
    for i, token in enumerate(array_rationale):
        if i == 0:
            if (token != 0):
                fragment["start_token"] = i
            continue
        if (token != 0) and (array_rationale[i-1] == 0):
            fragment["start_token"] = i
            continue
        if (token == 0) and (array_rationale[i-1] != 0):
            fragment["end_token"] = i
            list_rationales.append(fragment)
            fragment = {}
            continue
    if token != 0:
        fragment["end_token"] = i+1
        list_rationales.append(fragment)
    return list_rationales


if __name__=="__main__":
    model="G2X"
    data_path = "data"
    idx_to_file_path = os.path.join(data_path,"idx_to_file.pkl")
    # Read idx to id file
    with open(idx_to_file_path, 'rb') as f:
        idx_to_file = pkl.load(f)
    with open(os.path.join(data_path,"post_id_divisions.json"), 'rb') as f:
        post_id_division = json.load(f)
    # Read test set
    x_test = np.load(os.path.join(data_path, "x_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))
    rationales_test = np.load(os.path.join(data_path, "rationales_test.npy"))
    pred_rationales_test = np.load(os.path.join(data_path, f"x_test-{model}.npy"))
    print("Hello")
    test_files = idx_to_file['test']
    code_to_class = {0:"normal", 1:'offensive', 2:'hatespeech'}
    print("Creating dataset list.")
    dataset = []
    for idx in test_files:
        doc_id = test_files[idx]
        class_str = code_to_class[np.argmax(y_test[idx])]
        encoded_tokens = x_test[idx][np.where(x_test[idx]==1)[0][0]+1:].astype(int)
        inst_rationales = [rationales_test[idx][np.where(x_test[idx]==1)[0][0]+1:].astype(int)]
        str_labels = ['normal', 'offensive', 'hatespeech']
        dataset_instance = [doc_id, class_str, encoded_tokens, inst_rationales, str_labels]
        dataset.append(dataset_instance)
    print("Converting dataset to ERASER format.")
    convert_to_eraser_format(dataset, method="union", save_split=True, save_path="data/", id_division=post_id_division)
    #create eraser predictions format
    print("Converting predictions to ERASER format.")
    output_eraser_format(model)

    print("Finito.")