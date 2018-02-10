from bs4 import BeautifulSoup
from tqdm import tqdm

import urllib
import os


# specify the url
data_source = "https://www.cs.toronto.edu/~vmnih/data/"


def require_dir(path):
    '''Make sure that the path exists otherwise create the directory structure'''
    if not os.path.exists(os.getcwd() + path):
        try:
            os.makedirs(os.getcwd() + path)
        except:
            raise


def scraper(base_path):
    """

    Args:
        base_path: path to the dataset

    Returns:
        downloads the data that we need
    """
    with urllib.request.urlopen(base_path) as url:
        response = url.read()

    soup = BeautifulSoup(response, 'html.parser')

    # pull out just the html pages
    links = [tag['href'] for tag in soup.findAll('a') if ".html" in tag['href']]

    for link in links:

        print("Starting: {}".format(link))

        with urllib.request.urlopen(base_path+link) as img_idx_url:
            images_response = img_idx_url.read()

        images_soup = BeautifulSoup(images_response, 'html.parser')

        # pull out just the html pages
        img_links = [tag['href'] for tag in images_soup.findAll('a') if ".tif" in tag['href']]

        for img_link in tqdm(img_links):

            save_file_path = img_link.split('~vmnih')[1]

            # create the file directory if it doesn't exist and return back the location where it should be saved
            folder_path = os.path.dirname(save_file_path)
            require_dir(folder_path)

            urllib.request.urlretrieve(img_link, os.getcwd()+save_file_path)




