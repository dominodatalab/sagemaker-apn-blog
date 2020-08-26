import pickle 
import numpy as np

loaded_model = pickle.load(open('results/smallModel.pkl', 'rb'))

def predict(dropperc, mins, consecmonths, income):
    features=[]
    features.append(dropperc)
    features.append(mins)
    features.append(consecmonths)
    features.append(income)
    final = np.reshape(features, (1, -1))
    return list(loaded_model.predict_proba(final)[:,1])