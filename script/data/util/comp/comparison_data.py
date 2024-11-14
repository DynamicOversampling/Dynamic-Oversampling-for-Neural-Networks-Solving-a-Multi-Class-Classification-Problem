from script.configs.configs import ModelConfig
from script.configs.setup import DataFrameConfigs, configSMOTE
from script.utils import loadDF
from script.visualization.util.utils import parseEntries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def comparison():
    """
    function to generate comparison excel for KNN and RF for the model:
    ModelConfig(8, 3, 0.25)
    """
    results = []

    for index, configDf in enumerate(DataFrameConfigs):

        print(index)
        configMdd = ModelConfig(8, 3, 0.25)
        df = loadDF(index, "../../../../", configMdd=configMdd)

        f = df.drop('target', axis=1).values
        t = df['target'].values

        X_train, X_test, y_train, y_test = train_test_split(f, t, test_size=0.2, random_state=42)

        k_values = [i for i in range(1, 10)]
        cv_scores_knn = []
        std_scores_knn = []
        test_scores_knn = []
        f1_scores_knn = []

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)

            cv_score = cross_val_score(knn, X_train, y_train, cv=5)
            cv_scores_knn.append(np.mean(cv_score))
            std_scores_knn.append(np.std(cv_score))

            knn.fit(X_train, y_train)

            test_score = knn.score(X_test, y_test)
            test_scores_knn.append(test_score)

            y_pred_knn = knn.predict(X_test)
            f1 = f1_score(y_test, y_pred_knn, average='macro')
            f1_scores_knn.append(f1)

        max_cv_score_knn = max(cv_scores_knn)
        max_std_score_knn = std_scores_knn[cv_scores_knn.index(max(cv_scores_knn))]
        max_test_score_knn = max(test_scores_knn)
        max_f1_score_knn = max(f1_scores_knn)

        print(f"KNN Max CV Score: {max_cv_score_knn:.4f} ({max_std_score_knn:.4f})")
        print(f"KNN Max Test Score: {max_test_score_knn:.4f}")
        print(f"KNN Max F1 Score: {max_f1_score_knn:.4f}")

        n_estimators_values = list(range(1, 10))
        cv_scores_rf = []
        test_scores_rf = []
        std_scores_rf = []
        f1_scores_rf = []

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for n in n_estimators_values:
            rf = RandomForestClassifier(n_estimators=n)

            cv_score = cross_val_score(rf, X_train, y_train, cv=5)
            cv_scores_rf.append(np.mean(cv_score))
            std_scores_rf.append(np.std(cv_score))

            rf.fit(X_train, y_train)
            test_score = rf.score(X_test, y_test)
            test_scores_rf.append(test_score)

            y_pred_rf = rf.predict(X_test)
            f1_rf = f1_score(y_test, y_pred_rf, average='macro')
            f1_scores_rf.append(f1_rf)

        max_cv_score_rf = max(cv_scores_rf)
        max_std_score_rf = std_scores_rf[cv_scores_rf.index(max_cv_score_rf)]
        max_test_score_rf = max(test_scores_rf)
        max_f1_score_rf = max(f1_scores_rf)

        print(f"RF Max CV Score: {max_cv_score_rf:.4f} ({max_std_score_rf:.4f})")
        print(f"RF Max Test Score: {max_test_score_rf:.4f}")
        print(f"RF Max F1 Score: {max_f1_score_rf:.4f}")

        _, crossfold_data = parseEntries(configDf, configMdd, configSMOTE, 2, 1, "../../../../", enforce=False)
        training_accs = max(crossfold_data["TestAcc"]["mean"])
        training_accs_std = crossfold_data["TestAcc"]["std"][-1]
        training_f1 = max(crossfold_data["TestF1"]["mean"])
        training_f1_std = crossfold_data["TestF1"]["std"][-1]

        print(f"Training Acc: {training_accs:.4f} ({training_accs_std:.4f})")
        print(f"Training F1: {training_f1:.4f} ({training_f1_std:.4f})")

        results.append({
            'Index': index,
            'KNN Max CV Score': f"{max_cv_score_knn:.4f} ({max_std_score_knn:.4f})",
            'KNN Max Test Score': f"{max_test_score_knn:.4f}",
            'KNN Max F1 Score': f"{max_f1_score_knn:.4f}",
            'RF Max CV Score': f"{max_cv_score_rf:.4f} ({max_std_score_rf:.4f})",
            'RF Max Test Score': f"{max_test_score_rf:.4f}",
            'RF Max F1 Score': f"{max_f1_score_rf:.4f}",
            'Training Acc': f"{training_accs:.4f} ({training_accs_std:.4f})",
            'Training F1': f"{training_f1:.4f} ({training_f1_std:.4f})"
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel("model_results.xlsx", index=False)


comparison()
