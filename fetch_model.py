import csv
import pandas as pd
from .model import fetcher

PATH = "/model/model.h5"

def list_model(path):
    """Lists models from path provided.
    Args:
        path (_str_): path to the csv file containing list of models
    
    Returns:
        dict: A mapped dictionary of filename and url.
    """
    
    df = pd.read_csv(path)
    filenames = df.iloc[:, 1].values
    url = df.iloc[:, 1].values
    return dict(zip(filenames, url))


def model_save(m1, models, path=PATH):
    """Saves the model to the path.

    Args:
        m1 (_str_): is the chosen model from the user.
        models (_dir_): dictionary of models in form of <model>:<url>
    """
    
    print("fetching model:\t", model)
    fetcher.fetch_model(models[m1], PATH)
    print("Model saved at:\t", path)
    
    saved={m1:models[m1]}
    
    with open('model/saved_model.csv', 'w', newline='') as csvfile:
        labelwriter = csv.writer(csvfile)
        labelwriter.writerow(['model', 'url'])
        labelwriter.writerows(saved)
        
if __name__ == "__main__":
    
    choices = dict()
    models = list_model('models.csv')
    print("Please decide the model out of the following available:")
    ct = 1
    
    
    for model in models:
        choices[ct] = model
        print(f"Press {ct} for fetching {model}")
        ct+=1
    
       
    m1 = models[ct[int(input("Choose the model:\t"))]]
    
    model_save(m1, models)
    
    
    
    


    
