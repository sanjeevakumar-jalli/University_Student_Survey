from xgboost import XGBClassifier

def train_model(X_train, Y_train):
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, Y_train)
    return model