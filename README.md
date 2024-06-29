# Project Report Overview

## Introduction

This project involves two main components:
1. **Web Scraping and Data Analysis**: Automating the collection, cleaning, and analysis of text data from web pages.
2. **ChatBot Development**: Leveraging NLP techniques to create an interactive chatbot.

## Web Scraping

The provided Python script automates the process of web scraping, data cleaning, and text analysis.

### Key Libraries
- **BeautifulSoup**: For web scraping.
- **NLTK**: For natural language processing.
- **Scikit-learn**: For TF-IDF vectorization.

### Objective
To build a knowledge base from scraped web content by extracting relevant terms and sentences related to a specific topic, specifically Ratan Tata.

### Steps Involved
1. **Web Scraping and Data Collection**
   - Uses `requests` and `BeautifulSoup` to fetch and parse web pages.
   - Extracts and saves content from paragraph tags in text files.

2. **Data Cleaning**
   - Converts text to lowercase, removes punctuation, tokenizes, removes stopwords, and lemmatizes words.
   - Saves cleaned text for analysis.

3. **Feature Extraction and Analysis**
   - Utilizes TF-IDF vectorizer to identify and score relevant terms.
   - Aggregates scores to identify top terms.

4. **Sentence Extraction and Knowledge Base Construction**
   - Finds sentences containing top terms in the original content.
   - Constructs a knowledge base mapping terms to relevant sentences.

### Key Functions
- `scrape_and_save(urls)`: Scrapes content from URLs and saves text.
- `clean_text_files(input_directory)`: Cleans and saves text files.
- `extract_top_terms_from_all_files(directory, top_n=40)`: Identifies top N terms.
- `find_sentences_with_terms_in_content_files(directory, terms)`: Finds sentences for each term.

### Conclusion
The script efficiently processes web content to construct a valuable knowledge base for various applications.

## ChatBot System

The chatbot uses NLP techniques to interact dynamically with users.

### Key Components
1. **Spacy for Text Processing**
   - Extracts linguistic features like lemmatization and POS tagging.

2. **TF-IDF Vectorization for Response Selection**
   - Transforms text inputs into a TF-IDF matrix for identifying relevant responses.

3. **NLTK for Tokenization and Stopwords Removal**
   - Tokenizes user inputs and removes stopwords.

4. **Personalized User Modeling**
   - Updates user models based on interactions.

5. **Regex for Text Cleaning**
   - Cleans responses using regular expressions.

6. **Named Entity Recognition (NER)**
   - Uses Spacy's `en_core_web_sm` model to recognize entities in text.

### Dialogue Flow
1. **Greeting and User Identification**
   - Greets user and identifies or creates a new user model.

2. **Input Processing and Personal Information Extraction**
   - Analyzes inputs for personal details using NER.

3. **Preference Learning**
   - Infers likes and dislikes from user inputs.

4. **Response Generation**
   - Generates relevant responses using TF-IDF and cosine similarity.

5. **Feedback Loop for Preferences**
   - Adjusts user model based on reactions to previous responses.

### Evaluation and Analysis
**Strengths**:
- Personalization and flexibility.
- Scalability for additional data sources.

**Weaknesses**:
- Limited context understanding.
- Dependence on the quality of the knowledge base.

### User Feedback
Users find the chatbot useful for quickly retrieving relevant information and appreciate the GUI's modern feel. Suggestions include the ability to save chat conversations.

## How to Use the ChatBot

### Prerequisites
Ensure you have the following libraries installed:
- `requests`
- `beautifulsoup4`
- `nltk`
- `scikit-learn`
- `spacy`
- `tkinter` (for GUI)

### Setup
1. **Clone the Repository**
   ```sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
2. **Install Dependencies**

   ```pip install -r requirements.txt```

3. **Download Spacy Model**

  ```python -m spacy download en_core_web_sm```  

4. **Set Input Directory**
Update the input_directory variable in the script to match your machine's directory structure.

### Running the Script
  Run the Web Scraping Script

```sh
python web_scraping.py
This will scrape the data, clean it, and build the knowledge base.
```

Run the ChatBot

``` python chat_bot.py```
This will start the chatbot in the command line interface.

Run the ChatBot with GUI
To use the chatbot with a GUI, comment out the last line (start_chat_session()) in chat_bot.py and run:

``` python chat_bot_gui.py```


### Interacting with the ChatBot

- Start the bot, and it will greet you and ask for your user ID.
- If you are a returning user, it will load your previous session data.
- Ask questions related to Ratan Tata and the Tata Group, and the bot will provide responses based on the knowledge base.
- The bot will learn from your interactions, updating your user model with likes, dislikes, and personal information for personalized responses.

### Future Work
Improve context management.
Expand the knowledge base.
Refine entity recognition for better conversation quality.

- Knowledge Base
The knowledge base includes pivotal keywords related to the Tata Group and Ratan Tata.

- User Models
User models capture individual data, including name, likes, dislikes, and personal information, enabling personalized interactions.

- Sample User Models
User models are created or updated based on NER and user inputs.

- Survey Results
Users rate the chatbot positively for ease of interaction, relevance of responses, and personalization effectiveness.

### Conclusion
This project demonstrates the potential of NLP in creating interactive, personalized user experiences. Future enhancements will focus on context management, knowledge base expansion, and entity recognition improvements.
