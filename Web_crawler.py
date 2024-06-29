## Denish Kalariya
## DMK220001
## Project 1

from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from nltk.tokenize import sent_tokenize
import os
import nltk
nltk.download('punkt') 
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import pickle
import string


starter_url = "https://en.wikipedia.org/wiki/Ratan_Tata"
input_directory = "/Users/denish/Desktop/(Denish)/NLP/Project1_3"


## This function scrape the URls from the starter URL using sent_tokenize
## Also saves the URL content to txt files
def scrape_and_save(urls):
    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an exception for 4XX or 5XX status codes
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from paragraph tags as these are more likely to contain coherent sentences
            paragraphs = soup.find_all('p')
            text = ' '.join([para.get_text() for para in paragraphs])
            
            # Tokenize the text into sentences
            sentences = sent_tokenize(text)
            formatted_text = '\n'.join(sentences)  # Join sentences with newline characters
            
            # Creating a valid filename from the URL or using the index
            filename = os.path.join(input_directory, f"website_content_{i}.txt")  # Specify your path
            
            # Writing to file
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(formatted_text)
                
            print(f"Content saved to {filename}")
            
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
        except Exception as e:
            print(f"An error occurred with {url}: {e}")


### This functions cleans the text file extracted from the URLS using various NLP techniques
def clean_text_files(input_directory):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_directory, filename)
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Clean the text
            text = text.lower()
            text = ''.join([char for char in text if char not in string.punctuation])
            words = word_tokenize(text)
            words = [word for word in words if word not in stop_words]
            words = [lemmatizer.lemmatize(word) for word in words]
            cleaned_text = ' '.join(words)
            
            # Write the cleaned text to a new file
            cleaned_file_path = os.path.join(input_directory, f"{filename}_cleaned.txt")
            with open(cleaned_file_path, 'w', encoding='utf-8') as cleaned_file:
                cleaned_file.write(cleaned_text)
            
            print(f"Cleaned text saved to {cleaned_file_path}")


## this functions extract the top terms from all the texts using Tf-idf
def extract_top_terms_from_all_files(directory, top_n=25):
    # Ensure top_n is within the desired range
    top_n = max(25, min(top_n, 40))

    texts = []
    
    # Read each cleaned text file
    for filename in os.listdir(directory):
        if filename.endswith("_cleaned.txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())

    # Check if there are any texts to process
    if not texts:
        print("No cleaned text files found.")
        return
    
    # Compute TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names()
    
    # Aggregate TF-IDF scores for each term across all documents
    aggregated_scores = np.sum(tfidf_matrix, axis=0)
    scores = np.squeeze(np.asarray(aggregated_scores))
    
    # Get indices of top N scores
    top_indices = scores.argsort()[-top_n:][::-1]
    
    # Extract top terms
    top_terms = [feature_names[index] for index in top_indices]
    
    print(f"Top {top_n} terms across all documents are: {top_terms}")



## Functions call the terms and based on the terms it searches in the file and build the knowledge_base as dictionary of relavant sentences
def find_sentences_with_terms_in_content_files(directory, terms):
    sentences_dict = {term: [] for term in terms}
    # Adjusted regex pattern to match non-cleaned content files
    pattern = re.compile(r'website_content_\d+\.txt$')

    for filename in os.listdir(directory):
        if pattern.match(filename):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                sentences = sent_tokenize(text)

                for sentence in sentences:
                    for term in terms:
                        # Using word boundary regex to ensure exact word match
                        if re.search(r'\b' + re.escape(term) + r'\b', sentence, re.IGNORECASE):
                            sentences_dict[term].append(sentence)

    return sentences_dict




def main():

#     starter_url = "https://en.wikipedia.org/wiki/Ratan_Tata"
#     input_directory = "/Users/denish/Desktop/(Denish)/NLP/Project1_3"

#     r = requests.get(starter_url)
#     data = r.text
#     soup = BeautifulSoup(data, 'html.parser')

# # Filter only the links within the main content of the article
#     main_content = soup.find('div', id='bodyContent')  # or soup.find('div', {'id': 'mw-content-text'}) for more specific targeting

#     counter = 0
# # Write URLs to a file
#     with open('urls.txt', 'w') as f:
#         for link in main_content.find_all('a', href=True):
#             href = link['href']
#         # Filter out non-article and administrative links
#             if href.startswith('/wiki/') and not href.startswith(('/wiki/Special:', '/wiki/Help:', '/wiki/File:')):
#                 full_url = urljoin('https://en.wikipedia.org', href)
#                 print(full_url)
#                 f.write(full_url + '\n')
#                 counter += 1
#                 if counter > 20:  # Limit to first 20 relevant links
#                     break

#     print("End of crawler")
    

    r = requests.get(starter_url)
    data = r.text
    soup = BeautifulSoup(data, 'html.parser')

# Filter only the links within the main content of the article
    main_content = soup.find('div', id='bodyContent')  # or soup.find('div', {'id': 'mw-content-text'}) for more specific targeting

    counter = 0
# Write URLs to a file
    with open('urls.txt', 'w') as f:
        for link in main_content.find_all('a', href=True):
            href = link['href']
        # Filter out non-article and administrative links
            if href.startswith('/wiki/') and not href.startswith(('/wiki/Special:', '/wiki/Help:', '/wiki/File:')):
                full_url = urljoin('https://en.wikipedia.org', href)
                print(full_url)
                f.write(full_url + '\n')
                counter += 1
                if counter > 20:  # Limit to first 20 relevant links
                    break

    print("End of crawler")


    ## Write URLS in urls.txt
    with open('urls.txt', 'r') as f:
        urls = f.read().splitlines()
    for u in urls:
        print(u)


    ## Calls all the functions defined above
    scrape_and_save(urls)

    clean_text_files(input_directory)

    knowledge_base = extract_top_terms_from_all_files(input_directory, top_n=40)  # Example: extracting top 40 terms
    print(knowledge_base)


    ## manually defining the terms from the domain knowledge
    knowledge_base = ['tata', 'india', 'mumbai', 'indian', 'state', 'company', 'bombay', 'mistry', 'british', 'award', 'cornell', 'university', 'ratanji',  'maharashtra', 'family', 'chairman', 'steel', 'group', 'ratan']

    sentences_dict = find_sentences_with_terms_in_content_files(input_directory, knowledge_base)


    ## saving the knowledge_base in pickle file
    with open('knowledge_base.pkl', 'wb') as file:
        pickle.dump(sentences_dict, file)

    print("Knowledge base has been pickled and saved to 'knowledge_base.pkl'.")



## Calling the main function
if __name__ == "__main__":
    main()