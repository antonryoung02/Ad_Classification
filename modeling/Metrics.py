class PLMetrics:
    def __init__(self):
        self.train_acc = 0
        self.valid_acc = 0
        self.valid_precision = 0
        self.valid_recall = 0
        self.valid_f1 = 0
        self.valid_auroc = 0

    def get_metrics_dict(self):
        return {
        'accuracy': self.valid_acc,
        'precision': self.valid_precision,
        'recall': self.valid_recall,
        'f1': self.valid_f1,
        'auroc': self.valid_auroc,
    }

