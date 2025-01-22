from sklearn.metrics import confusion_matrix
import json

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