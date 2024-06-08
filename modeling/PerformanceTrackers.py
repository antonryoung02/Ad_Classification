from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
import numpy as np
import wandb

class PerformanceTracker:
    """Stores model predictions and provides diagnostic/performance plots"""
    def __init__(self):
        self.metrics = {"accuracy":[], "precision":[], "recall":[], "pr_auc":[], "f1":[]}
        self.threshold = 0.5

    def record_metrics(self, y_true, y_prob):
        y_pred = y_prob > self.threshold
        self.metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        self.metrics['precision'].append(precision_score(y_true, y_pred, average='binary'))
        self.metrics['recall'].append(recall_score(y_true, y_pred, average='binary'))
        self.metrics['pr_auc'].append(average_precision_score(y_true, y_prob))
        self.metrics['f1'].append(f1_score(y_true, y_pred, average='binary'))

    def get_avg_metrics(self):
        avg_metrics = {key: np.mean(values) for key, values in self.metrics.items()}
        return avg_metrics


class WandBPerformanceTracker:
    def record_metrics(self, y_true, y_prob, fold_name):
        y_pred = y_prob > 0.5
        accuracy = accuracy_score(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred, average='binary')
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)  
        specificity = tn / (tn + fp)  

        wandb.log({"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity, "pr_auc": pr_auc, "f1": f1})
