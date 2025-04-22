import os
import pickle
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.metrics import make_scorer, accuracy_score


class TrainModel:

    X: pd.DataFrame
    y: pd.Series
    SEED:int = 42

    def __init__(self):
        data = datasets.load_iris(as_frame=True, return_X_y=True)
        self.X = data[0]
        self.y = data[1]
    
    def train(self)->None:
        params = {
            'n_estimators': randint(50, 200),
            'max_depth': [None] + list(randint(5, 20).rvs(5)),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5),
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.SEED)

        scoring = {
            'accuracy': make_scorer(accuracy_score),
        }

        model = RandomForestClassifier(random_state=self.SEED)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=10,
            scoring=scoring,
            cv=cv,
            refit='accuracy',
            random_state=self.SEED,
            n_jobs=-1,
            return_train_score=True
        )

        random_search.fit(self.X, self.y)

        print("Best parameters found:", random_search.best_params_)
        print("Best cross-validation ROC AUC score:", random_search.best_score_)

        filename = 'app/model/best_rf_iris_model.pkl'
        directory = os.path.dirname(filename)

        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, 'wb') as file:
            pickle.dump(random_search.best_estimator_, file)


    
    def main(self) -> None:
        self.train()
    

if __name__ == '__main__':

    ml = TrainModel()
    ml.main()
