import gdown

def fetch_model(url, output):
    """
    Fetches a file from Google Drive and saves it to the specified path.
    
    Parameters:
    url (str): The sharable link to the file on Google Drive.
    output (str): The path where the file will be saved on the local device.
    """
    
    file_id = url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    gdown.download(download_url, output, quiet=False)
