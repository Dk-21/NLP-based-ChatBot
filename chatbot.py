### Denish Kalariya
## DMK220001
### Project 1
import random
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
import spacy 
import re

## Loading the model using Spacy 
nlp = spacy.load("en_core_web_sm")

# Load your knowledge base

## Loading the knowledge_base
with open('knowledge_base.pkl', 'rb') as file:
    knowledge_base = pickle.load(file)

# Example user model template
user_model_template = {
    'name': '',
    'personal_info': {},
    'likes': [],
    'dislikes': []
}

# Function to save user model
def save_user_model(user_id, user_model):
    filename = f'user_model_{user_id}.json'
    with open(filename, 'w') as file:
        json.dump(user_model, file)

# Function to load user model
def load_user_model(user_id):
    filename = f'user_model_{user_id}.json'
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    else:
        return user_model_template.copy()


## Prepocess the text to remove unnecessary stuff
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + list(punctuation))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens




## Get the keywords from the response for like and dislike
def get_keywords(text):
    doc = nlp(text)
    # Initialize an empty list to hold potential keywords
    potential_keywords = []

    # Check for verb like "like" or "dislike" and capture the object
    for token in doc:
        if token.lemma_ in ["like", "dislike"] and token.dep_ == "ROOT":
            for child in token.children:
                ## extracting nouns and adjective from the response
                if child.dep_ in ["dobj", "prep"]:
                    potential_keywords.extend([child.text] + [grandchild.text for grandchild in child.children if grandchild.dep_ == "pobj"])
                    break

    # If no keywords found through verbs, fallback to named entities and noun chunks
    if not potential_keywords:
        named_entities = [ent.text for ent in doc.ents]
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        potential_keywords = named_entities + noun_chunks

    # Deduplicate while preserving order
    seen = set()
    deduplicated_keywords = [x for x in potential_keywords if not (x in seen or seen.add(x))]

    # Prefer longer phrases as they are more likely to be specific
    deduplicated_keywords.sort(key=lambda x: len(x), reverse=True)

    return deduplicated_keywords[0] if deduplicated_keywords else None


def preprocess_response(text):
    # Pattern to match numbers in square brackets (e.g., "[123]")
    pattern = r'\[\d+\]'
    # Replace matched patterns with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


entity_labels = [
    'PERSON',  # People, including fictional
    'GPE',     # Countries, cities, states
    'ORG',     # Companies, agencies, institutions
    'DATE',    # Absolute or relative dates or periods
    'LOC',     # Non-GPE locations, mountain ranges, bodies of water
    'PRODUCT', # Objects, vehicles, foods, not services
    'EVENT',   # Named hurricanes, battles, wars, sports events, etc.
    'WORK_OF_ART', # Titles of books, songs, etc
    'LANGUAGE', # Any named language
    'FAC'      # Buildings, airports, highways, bridges, etc.
]


## Extract the person Information using NER
def extract_personal_info(text):
    doc = nlp(text)
    personal_info = {}
    for ent in doc.ents:
        if ent.label_ in entity_labels:
            personal_info[ent.label_] = ent.text
    return personal_info


## Update the User model
def update_personal_info(user_id, personal_info):
    user_model = load_user_model(user_id)
    for key, value in personal_info.items():
        # Assuming 'personal_info' is a dictionary in the user model
        if key in user_model['personal_info']:
            # Avoid duplication or consider updating the info
            if value not in user_model['personal_info'][key]:
                user_model['personal_info'][key].append(value)
        else:
            user_model['personal_info'][key] = [value]
    save_user_model(user_id, user_model)



# Function to update user model with new information
def update_user_model(user_id, category, value):
    user_model = load_user_model(user_id)
    if category in ['likes', 'dislikes']:
        user_model[category].append(value)
    else: # For 'name' or 'personal_info'
        user_model[category] = value
    save_user_model(user_id, user_model)

# Function to tokenize and clean text
def preprocess_text(text):
    return word_tokenize(text.lower())

# Greetings
greetings_input = [
    'hi', 'hello', 'hey', 
    'greetings', 'good morning', 'good afternoon', 
    'good evening', 'hi there', 'hello there', 
    'hey there', 'what\'s up', 'sup', 
    'howdy', 'yo'
]

greetings_response = [
    'Hi there!', 'Hello!', 'Hey!', 
    'Greetings!', 'Good to see you!', 'Hi, how can I help?', 
    'Hello there!', 'Hey there! How are you?', 'What\'s up?', 
    'Howdy!', 'Yo!', 'How\'s it going?', 
    'Good day to you!', 'How can I assist you today?'
]


## For the response of the greeting
def get_greeting(sentence):
    for word in sentence.split():
        if word.lower() in greetings_input:
            return random.choice(greetings_response)

# Simple retrieval-based response
def get_response(user_id, user_input):
    # Flatten the knowledge base text into a list of sentences
    all_sentences = []
    for text in knowledge_base.values():
        if isinstance(text, str):  # Check if the value is a string
            all_sentences.extend(sent_tokenize(text))
        elif isinstance(text, list):  # Example: if the value might be a list of strings
            joined_text = ' '.join(text)  # Join the list into a single string
            all_sentences.extend(sent_tokenize(joined_text))
        # Add more conditions here if there are other data types in your knowledge base

    # Include the user's input for vectorization
    all_text = all_sentences + [user_input]
    
    # Vectorize the text using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_text)
    
    # Calculate cosine similarity between the user's input and all sentences
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Find the index of the highest similarity score
    max_sim_idx = np.argmax(cosine_sim)
    
    # If the highest score is low, indicate a lack of understanding

    ## therosold value can not be greater than 0.2
    if cosine_sim[max_sim_idx] < 0.2:  # Threshold can be adjusted
        return "I'm not sure how to respond to that."
    else:
        # Return the most similar sentence
        return all_sentences[max_sim_idx]



# Assuming each entry in the knowledge base might be related to a specific topic
# This could be extended to use NLP for extracting topics from user inputs

# Placeholder for the last topic discussed
last_topic = ""



def update_last_topic_from_input(user_input):
    global last_topic
    # This is a placeholder function. In practice, you would use NLP to extract topics from user_input
    # For simplicity, let's assume the user input directly mentions the topic for now
    words = user_input.split()
    if words:
        last_topic = words[-1]  # Update this logic based on actual topic extraction

## Empty string to dynamically store the last response of the chatbot
last_bot_response = ""

## get the last response and store in the empty string
def update_last_bot_response(response):
    global last_bot_response
    last_bot_response = response


## handle the likes and dislikes of the user
def handle_likes_dislikes(user_id, user_input):
    global last_topic
    if "i liked" or "i like" in user_input.lower():
        if last_topic:
            update_user_model(user_id, 'likes', last_topic)
            response = f"I'm glad to hear you liked {last_topic}! I've added it to your likes."
        else:
            response = "I'm not sure what you're referring to. Could you be more specific?"
    elif "i disliked" or "i dislike" in user_input.lower() or "don't like" in user_input.lower():
        if last_topic:
            update_user_model(user_id, 'dislikes', last_topic)
            response = f"I see, you don't like {last_topic}. I've added it to your dislikes."
        else:
            response = "I'm not sure what you're referring to. Could you be more specific?"
    else:
        response = None
    return response



## Extract the nouns and ADjectives form the response
def extract_nouns_and_adjectives(text):
    doc = nlp(text)
    # Extract nouns and adjectives from the text
    return [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]



## the main Chat function which generates the response from the chatbot for each query
def chat(user_id):
    global last_bot_response  # Ensure this is declared at the top of your script
    user_model = load_user_model(user_id)
    if not user_model['name']:
        user_name = input("Chatbot: Hi there! What's your name?\nYou: ")
        update_user_model(user_id, 'name', user_name)
    else:
        user_name = user_model['name']
        print(f"Chatbot: Welcome back, {user_name}! How can I help you today?")
    
    while True:
        user_input = input(f"{user_name}: ")
        if user_input.lower() == 'quit':
            print(f"Chatbot: Bye {user_name}! Have a great day!")
            break
        # Extract and update personal information
        personal_info = extract_personal_info(user_input)
        if personal_info:
            update_personal_info(user_id, personal_info)
            print("Chatbot: Thanks for sharing that with me!")

        greeting = get_greeting(user_input)
        if greeting is not None:
            print(f"Chatbot: {greeting} {user_name}!")

        elif "dislike" in user_input.lower() or "disliked" in user_input.lower() or "don't like" in user_input.lower():
            disliked_items = extract_nouns_and_adjectives(last_bot_response)
            if disliked_items:  # Ensure there are items to update
                    for item in disliked_items:
                        update_user_model(user_id, 'dislikes', item)
                    print(f"Chatbot: Noted. You don't like {', '.join(disliked_items)}.")

        elif "like" in user_input.lower() or "liked" in user_input.lower():
            liked_items = extract_nouns_and_adjectives(last_bot_response)
            if liked_items:  # Ensure there are items to update
                for item in liked_items:
                    update_user_model(user_id, 'likes', item)
                print(f"Chatbot: Noted. You like {', '.join(liked_items)}.")
        else:
            response = get_response(user_id, user_input)
            print(f"Chatbot: {preprocess_response(response)}")
    # Update the last_bot_response with the current bot's response
            last_bot_response = response




## working as main function which calls the main function and calls the chat function
def start_chat_session():
    print("Welcome to the Chatbot Service!")
    user_id = input("Please enter your user ID or type 'new' to create a new session:\n")
    
    if user_id.lower() == 'new':
        # Here you could generate a new unique user ID or ask the user to input a new ID
        # For simplicity, let's just ask for a user-defined ID
        new_user_id = input("Enter a new user ID to start:\n")
        chat(new_user_id)
    else:
        chat(user_id)

# Start the chat session
## To use the GUI please comment the below line
start_chat_session()

