import os
import csv
import pandas as pd
import model.fetcher as fetcher


DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR, "model", "model.h5")  # Use os.path.join for better compatibility



def list_model(path):
    """Lists models from path provided.
    Args:
        path (str): path to the CSV file containing the list of models.
    
    Returns:
        dict: A mapped dictionary of filename and URL.
    """

    df = pd.read_csv(path)
    models = df.iloc[:, 0].values
    url = df.iloc[:, 1].values
    return dict(zip(models, url))


def model_save(m1, models, path=MODEL_PATH):
    """Saves the model to the path.

    Args:
        m1 (str): The chosen model from the user.
        models (dict): Dictionary of models in the form <model>:<url>.
        path (str): Path where the model will be saved.
    """
    
    print("Fetching model:\t", m1)
    fetcher.fetch_model(models[m1], path)
    print("Model saved at:\t", path)
    
    saved = {m1: models[m1]}
    
    with open(os.path.join(DIR, 'model', 'saved_model.csv'), 'w', newline='') as csvfile:
        labelwriter = csv.writer(csvfile)
        labelwriter.writerow(['model', 'url'])
        labelwriter.writerows(saved.items())
        
        
def fetch_model():
    choices = dict()
    models = list_model(os.path.join(DIR, "model.csv",))
    print("Please decide the model out of the following available:\t")
    
    ct = 1
    for model in models:
        choices[ct] = model
        print(f"Press {ct} for fetching {model}")
        ct += 1
    
    m1 = choices[int(input("\nChoose the model:\t"))]
    model_save(m1, models)


if __name__ == "__main__":
    fetch_model()
