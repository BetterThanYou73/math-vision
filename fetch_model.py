import os
import csv
import pandas as pd
import model.fetcher as fetcher
import model.update_models_database as csv_update

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR, "model", "model.h5")  # Use os.path.join for better compatibility
MODELS_LIST="models.xlsx"

# ANSI escape code
green_text = "\033[92m"
blue_text = '\033[94m'
reset_text = "\033[0m"

def list_model(path):
    """Lists models from path provided.
    Args:
        path (str): path to the CSV file containing the list of models.
    
    Returns:
        dict: A mapped dictionary of filename and URL.
    """

    df = pd.read_excel(path)
    models = dict(zip(df['model'], zip(df['url'], df['description'])))
    return models


def model_save(m1, models, path=MODEL_PATH):
    """Saves the model to the path.

    Args:
        m1 (str): The chosen model from the user.
        models (dict): Dictionary of models in the form <model>:<url>.
        path (str): Path where the model will be saved.
    """
    
    print(f"{green_text}Fetching model:\t{blue_text}", m1,'\n')
    print(green_text)
    fetcher.fetch_model(models[m1][0], path)
    print(f"\n{green_text}Model saved at:\t{blue_text}", path)
    
    saved = {m1: models[m1]}
    
    with open(os.path.join(DIR, 'model', 'current_model.csv'), 'w', newline='') as csvfile:
        labelwriter = csv.writer(csvfile)
        labelwriter.writerow(['model', 'url', 'description'])
        labelwriter.writerows(saved.items())
        
        
def fetch_model():

    choices = dict()
    csv_update.fetch_csv(output=MODELS_LIST)
    models = list_model(os.path.join(DIR, MODELS_LIST))
    
    # Calculate maximum lengths for each column
    max_no_len = max(len(f"Press {ct}") for ct in range(1, len(models) + 1)) + 1
    max_model_len = max(len(model) for model in models) + 2
    max_desc_len = max(len(desc) for _, desc in models.values()) + 2
    
    # Print header with dynamic spacing
    print(f"{green_text}{'No.':<{max_no_len}} {'Model Name':<{max_model_len}} {'Description':<{max_desc_len}}{reset_text}")
    print(f"{green_text}{'-' * max_no_len} {'-' * max_model_len} {'-' * max_desc_len}{reset_text}")
    
    ct = 1
    for model, (url, desc) in models.items():
        choices[ct] = model
        print(f"{green_text}Press {ct:<{len('No.')}} {model:<{max_model_len}} {desc:<{max_desc_len}}{reset_text}")
        ct += 1
    
    m1 = choices[int(input(f"\n{green_text}Choose the model:\t{blue_text}"))]
    print(reset_text)
    model_save(m1, models)


if __name__ == "__main__":
    fetch_model()
