from bert_score import score
from nlgmetricverse import NLGMetricverse, load_metric
import pickle, os, json

## calculate scores
def calculate_scores(predictions, gts):
    metrics = [
        load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
        load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
        load_metric("rouge"),
        load_metric("cider"),
    ]
    Evaluator = NLGMetricverse(metrics)

    ## need to convert predictions and gts to list to fit with bert_score
    ### make sure predictions and gts are in the same order
    predictions = {k: v for k, v in sorted(predictions.items()) if k in gts}
    gts = {k: v for k, v in sorted(gts.items()) if k in predictions}

    predictions_list = list(predictions.values())
    gts_list = list(gts.values())

    for index in range(len(predictions_list)):
        predictions_list[index] = [predictions_list[index]]

    scores = Evaluator(predictions=predictions_list, references=gts_list)
    score_results = {}
    score_results["bleu_1"] = scores["bleu_1"]['score']
    score_results["bleu_4"] = scores["bleu_4"]['score']
    score_results["rouge"] = scores["rouge"]['rougeL']
    score_results["cider"] = scores["cider"]['score']

    return score_results

def gts():
    pkl_file = '/home/peihsin/Sinica/dataset/output_test_label_para6.pkl'
    groud_truth = {}
    with open(pkl_file, 'rb') as f:
        data_list = pickle.load(f)
        for item in data_list:
            if item['video_name'] == 'standard':
                continue
            groud_truth[item['video_name']] = item['labels']
    return groud_truth

def main():
    groud_truth = gts()
    All_file = {}
    folder_path = "/home/peihsin/Sinica/emnlp"
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json') and file_name.startswith('e'):
            file_path = os.path.join(folder_path, file_name)
            predictions = {}
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                for k, v in json_data.items():
                    if k == 'standard':
                        continue
                    predictions[k] = v
                All_file[file_name] = calculate_scores(predictions, groud_truth)

    with open("result_tommyB_list.json", "w") as f:
        json.dump(All_file, f)

if __name__ == "__main__":
    main()
