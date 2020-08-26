import pickle 
import numpy as np

loaded_model = pickle.load(open('results/gb1.pkl', 'rb'))

def predict(dropperc, mins, consecmonths, income):
    features=[]
    features.append(dropperc)
    features.append(mins)
    features.append(consecmonths)
    features.append(income)
    final = np.reshape(features, (1, -1))
    # Call the model's predict_proba() function with the input features
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier.predict_proba
    return list(loaded_model.predict_proba(final)[:,1])
