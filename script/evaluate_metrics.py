# for model and task
#     set p1
#     cms = []
#     per_seed_metrics = []
#     p1/0/res.csv
#     p1/1/res.csv
#     glob -> p1/*/res.csv returns a list of paths
#     for path in list of paths
#         read list csv
#         filter on target_task_answered == True
#         select y_true
#         select y_pred
#         classification report sklearn
#         compute cm sklearn with norm = true
#         append in cms
#         append in per_seed_metrics
#     mean and std on per_seed_metrics
#     np.mean(confusion_matrices, axis=0)
#     load  mean and std in a csv

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

base_path = "../../data/financial"

