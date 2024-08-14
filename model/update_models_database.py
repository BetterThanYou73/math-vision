import gdown

def fetch_csv(output, url="https://docs.google.com/spreadsheets/d/1iJooDOSVFjV0ryB8D87yvgf46fm4UJGDErheevsbkGw/edit?usp=drive_link"):
    """
    Fetches csv file containing list of models and their Google Drive links.
    
    Parameters:
    url (str): The sharable link to the file on Google Drive.
    output (str): The path where the file will be saved on the local device.
    """
    
    file_id = url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    gdown.download(download_url, output, quiet=True)