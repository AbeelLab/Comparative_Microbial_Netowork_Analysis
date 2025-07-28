import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
import optuna
from scipy import stats
from joblib import Parallel, delayed
import warnings
from pathlib import Path
from collections import defaultdict
from src.plotting import plot_confusion_matrices
from tqdm import tqdm
from contextlib import contextmanager
import joblib

warnings.filterwarnings('ignore')

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallBack(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallBack
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

class UniformDummyClassifier:
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        np.random.seed(self.random_state)
        n_samples = len(X)
        n_classes = len(self.classes_)
        predictions_per_class = n_samples // n_classes
        remainder = n_samples % n_classes

        predictions = []
        for cls in self.classes_:
            predictions.extend([cls] * predictions_per_class)
        if remainder:
            predictions.extend(np.random.choice(self.classes_, remainder, replace=True))
        np.random.shuffle(predictions)
        return np.array(predictions)

def load_data(raw_dir: Path, output_dir: Path):
    species_file = raw_dir / 'new_GTDB_species_relative_abundance_filterd_low_clr.xlsx'
    GTDB_species_id_name_dict_file = raw_dir / 'GTDB_species_id_name_dict.pkl'
    meta_data_file = raw_dir / 'metadata.tsv'

    with open(GTDB_species_id_name_dict_file, 'rb') as f:
        GTDB_species_id_name_dict = pickle.load(f)
    GTDB_species_id_name_dict = {str(k): v for k, v in GTDB_species_id_name_dict.items()}
    species_count = pd.read_excel(species_file, index_col=0)
    species_count.columns = species_count.columns.astype(str)
    species_count = species_count.rename(columns=GTDB_species_id_name_dict)
    meta_data = pd.read_csv(meta_data_file, sep='\t', index_col=0)

    with open(output_dir / 'keystone.pkl', 'rb') as file:
        keystone_dict = pickle.load(file)

    from src.keystone import get_all_species
    keystone = get_all_species(keystone_dict)

    cecum_meta_data = meta_data[meta_data['GUT_SECTION'] == 'Cecum']
    cecum_meta_data = cecum_meta_data[cecum_meta_data['TREATMENT'].isin(['CTR', 'AGP', 'PFA'])]
    cecum_meta_data = cecum_meta_data[cecum_meta_data['AGE'].isin([14, 21, 35])]
    cecum_species_count = species_count.loc[cecum_meta_data.index.tolist()]
    cecum_species_count = cecum_species_count[keystone]
    X = cecum_species_count.join(cecum_meta_data[['TYPE', 'AGE']])
    y = cecum_meta_data['TREATMENT'].tolist()
    X = pd.get_dummies(X, drop_first=False)
    feature_names = X.columns.tolist()
    X = X.to_numpy()
    y = np.array(y)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le, feature_names

def process_fold(fold_data, inner_cv, n_trials=30):
    fold_idx, train_idx, test_idx, X, y_encoded, le = fold_data

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    feature_selector = SelectKBest(mutual_info_classif, k=min(15, X_train.shape[1]))
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    X_test_selected = feature_selector.transform(X_test)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=1)
        scores = []
        for train_idx_inner, val_idx_inner in inner_cv.split(X_train_selected, y_train):
            X_inner_train, X_inner_val = X_train_selected[train_idx_inner], X_train_selected[val_idx_inner]
            y_inner_train, y_inner_val = y_train[train_idx_inner], y_train[val_idx_inner]
            model.fit(X_inner_train, y_inner_train)
            scores.append(f1_score(y_inner_val, model.predict(X_inner_val), average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params

    rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=1)
    rf_model.fit(X_train_selected, y_train)
    dummy_model = UniformDummyClassifier(random_state=42)
    dummy_model.fit(X_train_selected, y_train)

    rf_pred = rf_model.predict(X_test_selected)
    dummy_pred = dummy_model.predict(X_test_selected)

    train_dist = {le.inverse_transform([cls])[0]: count / len(y_train) for cls, count in pd.Series(y_train).value_counts().items()}
    test_dist = {le.inverse_transform([cls])[0]: count / len(y_test) for cls, count in pd.Series(y_test).value_counts().items()}
    dummy_dist = {le.inverse_transform([cls])[0]: count / len(dummy_pred) for cls, count in pd.Series(dummy_pred).value_counts().items()}

    return {
        'fold_idx': fold_idx,
        'best_params': best_params,
        'best_score': study.best_value,
        'rf_metrics': {
            'f1_macro': f1_score(y_test, rf_pred, average='macro'),
            'precision_macro': precision_score(y_test, rf_pred, average='macro'),
            'recall_macro': recall_score(y_test, rf_pred, average='macro'),
            'balanced_accuracy': balanced_accuracy_score(y_test, rf_pred)
        },
        'dummy_metrics': {
            'f1_macro': f1_score(y_test, dummy_pred, average='macro'),
            'precision_macro': precision_score(y_test, dummy_pred, average='macro'),
            'recall_macro': recall_score(y_test, dummy_pred, average='macro'),
            'balanced_accuracy': balanced_accuracy_score(y_test, dummy_pred)
        },
        'rf_per_class': {
            'f1': f1_score(y_test, rf_pred, average=None, zero_division=0),
            'precision': precision_score(y_test, rf_pred, average=None, zero_division=0),
            'recall': recall_score(y_test, rf_pred, average=None, zero_division=0)
        },
        'dummy_per_class': {
            'f1': f1_score(y_test, dummy_pred, average=None, zero_division=0),
            'precision': precision_score(y_test, dummy_pred, average=None, zero_division=0),
            'recall': recall_score(y_test, dummy_pred, average=None, zero_division=0)
        },
        'confusion_matrix_rf': confusion_matrix(y_test, rf_pred, labels=rf_model.classes_),
        'confusion_matrix_dummy': confusion_matrix(y_test, dummy_pred, labels=rf_model.classes_),
        'model_classes': rf_model.classes_,
        'train_dist': train_dist,
        'test_dist': test_dist,
        'dummy_dist': dummy_dist
    }


def run_nested_cv(X, y_encoded, le, figures_dir: Path, output_dir: Path, n_jobs=-1):
    summary_path = output_dir / 'rf_summary.txt'
    if summary_path.exists():
        print("RF results already exist, skipping computation.")
        return None, None, None
    
    print("Running cross-validation for RF classification...")
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_data = [(i, train_idx, test_idx, X, y_encoded, le) for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_encoded))]
    with tqdm_joblib(tqdm(total=len(fold_data), desc="Processing folds")):
        results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(process_fold)(data, inner_cv) for data in fold_data)

    rf_scores = {m: [] for m in ['f1_macro', 'precision_macro', 'recall_macro', 'balanced_accuracy']}
    dummy_scores = {m: [] for m in ['f1_macro', 'precision_macro', 'recall_macro', 'balanced_accuracy']}
    per_class_rf = {m: {cls: [] for cls in le.classes_} for m in ['f1', 'precision', 'recall']}
    per_class_dummy = {m: {cls: [] for cls in le.classes_} for m in ['f1', 'precision', 'recall']}
    best_params_list = []
    cm_rf, cm_dummy = [], []
    train_dists, test_dists, dummy_dists = [], [], []

    for result in results:
        for metric in rf_scores:
            rf_scores[metric].append(result['rf_metrics'][metric])
            dummy_scores[metric].append(result['dummy_metrics'][metric])

        best_params_list.append(result['best_params'])
        cm_rf.append(result['confusion_matrix_rf'])
        cm_dummy.append(result['confusion_matrix_dummy'])
        train_dists.append(result['train_dist'])
        test_dists.append(result['test_dist'])
        dummy_dists.append(result['dummy_dist'])

        for cls_idx, cls in enumerate(result['model_classes']):
            class_name = le.inverse_transform([cls])[0]
            for metric in ['f1', 'precision', 'recall']:
                per_class_rf[metric][class_name].append(result['rf_per_class'][metric][cls_idx])
                per_class_dummy[metric][class_name].append(result['dummy_per_class'][metric][cls_idx])

    print("Generating RF plots and saving results...")
    plot_confusion_matrices(cm_rf, cm_dummy, le, figures_dir)
    
    with open(summary_path, 'w') as f:
        metrics_df = pd.DataFrame({
            'Metric': ['F1-Score', 'Precision', 'Recall', 'Balanced Accuracy'],
            'RF Mean': [np.mean(rf_scores[m]) for m in ['f1_macro', 'precision_macro', 'recall_macro', 'balanced_accuracy']],
            'RF Std': [np.std(rf_scores[m]) for m in ['f1_macro', 'precision_macro', 'recall_macro', 'balanced_accuracy']],
            'Dummy Mean': [np.mean(dummy_scores[m]) for m in ['f1_macro', 'precision_macro', 'recall_macro', 'balanced_accuracy']],
            'Dummy Std': [np.std(dummy_scores[m]) for m in ['f1_macro', 'precision_macro', 'recall_macro', 'balanced_accuracy']]
        }).round(4)
        f.write("Performance Summary:\n")
        f.write(metrics_df.to_string(index=False) + "\n\n")

        per_class_df = pd.DataFrame({
            'Class': le.classes_,
            'F1 Mean (RF)': [np.mean(per_class_rf['f1'][cls]) for cls in le.classes_],
            'F1 Std (RF)': [np.std(per_class_rf['f1'][cls]) for cls in le.classes_],
            'Precision Mean (RF)': [np.mean(per_class_rf['precision'][cls]) for cls in le.classes_],
            'Precision Std (RF)': [np.std(per_class_rf['precision'][cls]) for cls in le.classes_],
            'Recall Mean (RF)': [np.mean(per_class_rf['recall'][cls]) for cls in le.classes_],
            'Recall Std (RF)': [np.std(per_class_rf['recall'][cls]) for cls in le.classes_],
            'F1 Mean (Dummy)': [np.mean(per_class_dummy['f1'][cls]) for cls in le.classes_],
            'F1 Std (Dummy)': [np.std(per_class_dummy['f1'][cls]) for cls in le.classes_],
        }).round(4)
        f.write("Per-Class Performance:\n")
        f.write(per_class_df.to_string(index=False) + "\n\n")

        avg_cm_rf = np.mean(cm_rf, axis=0)
        avg_cm_dummy = np.mean(cm_dummy, axis=0)
        f.write("Random Forest Confusion Matrix:\n")
        f.write(pd.DataFrame(avg_cm_rf.round(2), index=[f"True {c}" for c in le.classes_], columns=[f"Pred {c}" for c in le.classes_]).to_string() + "\n\n")
        f.write("Dummy Classifier Confusion Matrix:\n")
        f.write(pd.DataFrame(avg_cm_dummy.round(2), index=[f"True {c}" for c in le.classes_], columns=[f"Pred {c}" for c in le.classes_]).to_string() + "\n\n")

        rf_acc = np.trace(avg_cm_rf) / np.sum(avg_cm_rf)
        dummy_acc = np.trace(avg_cm_dummy) / np.sum(avg_cm_dummy)
        f.write(f"Accuracy: RF = {rf_acc:.4f}, Dummy = {dummy_acc:.4f}, Improvement = {rf_acc - dummy_acc:.4f}\n\n")

        f.write("Statistical Tests:\n")
        for metric in rf_scores:
            t_stat, p_val = stats.ttest_rel(rf_scores[metric], dummy_scores[metric])
            p_val_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
            f.write(f"{metric} t-test: t={t_stat:.4f}, p={p_val_one_sided:.4f}, Significant: {'Yes' if p_val_one_sided < 0.05 else 'No'}\n")

        f.write("\nPer-Class t-tests:\n")
        for cls in le.classes_:
            f.write(f"\nClass {cls}:\n")
            for metric in ['f1', 'precision', 'recall']:
                t_stat, p_val = stats.ttest_rel(per_class_rf[metric][cls], per_class_dummy[metric][cls])
                p_val_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
                f.write(f"  {metric}: t={t_stat:.4f}, p={p_val_one_sided:.4f}, Significant: {'Yes' if p_val_one_sided < 0.05 else 'No'}\n")

    return rf_scores, dummy_scores, best_params_list
