from sklearn.metrics import confusion_matrix
from transformers import pipeline
import json

summarizer = pipeline("summarization", model="./pretrained_models/bart-large-cnn/", device=0)
classifier = pipeline("text-classification", 
                      model="./pretrained_models/distilbert-base-uncased-finetuned-sst-2-english/", 
                      device=0)

def classify_text(text):
    yes_keywords = ["yes"]
    no_keywords = ["no"]
    
    text_lower = text.lower()
    
    yes_score = sum(keyword in text_lower for keyword in yes_keywords)
    no_score = sum(keyword in text_lower for keyword in no_keywords)
    
    if yes_score > no_score:
        return "Yes"
    elif no_score > yes_score:
        return "No"
    else:
        # 将长文本分割为句子
        words = text.split(" ")

        if(len(words) > 300):
            words = words[:300]
            text = " ".join(words)

        if len(words) > 100:
            summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            result = classifier(summary)[0]
        else:
            result = classifier(text)[0]

        return "Yes" if result['label'] == "POSITIVE" else "No" if result['label'] == "NEGATIVE" else "Neutral"

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return 100 * accuracy, 100 * precision, 100 * recall, 100 * f1

def calculate_displines(y_true, y_pred):
    with open("./data/disciplines.json", encoding="utf-8") as file:
        disciplines = json.load(file)
    
    data = {}

    for dis in ["CliMed", "MoleBioGene", "BioBioChem", "MultiDis", "PharToxi", "NeuBeha", "Chem", "Immu", "Micro"]:
        accu, prec, rec, f1 = calculate_metrics(y_true[disciplines[dis]], y_pred[disciplines[dis]])

        data[dis] = {
            "accu": accu,
            "prec": prec,
            "rec": rec,
            "f1": f1
        }

    return data