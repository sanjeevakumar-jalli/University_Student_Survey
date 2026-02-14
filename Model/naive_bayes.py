from sklearn.naive_bayes import GaussianNB

def train_model(X_train, Y_train):
    model = GaussianNB()
    model.fit(X_train, Y_train)
    return model