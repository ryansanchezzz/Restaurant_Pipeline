Restaurant Happy Hour URL Predictor

___Overview___
This project builds a machine learning model that predicts the correct sub-page of a restaurant website containing happy hour information.
Many restaurant websites contain multiple pages (menus, events, promotions, etc.). This project attempts to automatically identify which sub-URL 
contains the relevant happy hour details.

The pipeline consists of two main components:
Web Scraping Pipeline
 * Extracts links from restaurant base URLs
 * Scrapes webpage text or uses OCR for PDFs/images
 * Labels the correct happy hour page
Machine Learning Model
 * Converts webpage text and URL features into TF-IDF vectors
 * Trains a feed-forward neural network (PyTorch)
 * Predicts the most likely sub-page containing happy hour information


___Note on Scraper___
The full web scraping pipeline used to generate the dataset was developed collaboratively with a teammate.
Some scraping utilities and tools are not included in this repository.

___Dataset Generation___
The scraper performs the following steps:
1. Load restaurant base URLs from a CSV file
2. Extract all links from the base webpage
3. Scrape page text using an AI scraping tool
4. Use OCR when the content is a PDF or image
5. Label the correct happy hour page

___Feature Engineering___
Two feature types are used:
1. Text Features
TF-IDF vectorization of scraped webpage content.

2. URL Features
Character-level TF-IDF features extracted from sub-URLs.

These features are combined into a single vector for each candidate page.

___Dependencies___
Install required packages:
pip install -r requirements.txt

Main dependencies:
PyTorch
NumPy
Scikit-learn
BeautifulSoup
aiohttp
