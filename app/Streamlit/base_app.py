"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
#dependencies
import streamlit as st
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib,os
import nltk
import pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import unicodedata
import string
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import nltk
download_dir = os.path.join(os.getcwd(), 'data', 'nltk_data')

# Create the directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Add this directory to the NLTK data path
nltk.data.path.append(download_dir)

# Download necessary resources
nltk.download('punkt', download_dir=download_dir)
nltk.download('stopwords', download_dir=download_dir)
nltk.download('wordnet', download_dir=download_dir)
nltk.download('omw-1.4', download_dir=download_dir)

# Plotting Function
# Visualize Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(4, 3))  # Create figure and axis
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmap, xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)  # Pass the figure to st.pyplot()

# The main function where we will build the actual app
def main():
	"""News Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("News Classifer")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options =  ["Project Overview", "Importing Packages", "Loading Data", "Initial Data Inspection", "Data Cleaning","EDA","Data Preparation","Model Building, Training, Tuning and Testing","Prediction"]
	section = st.sidebar.selectbox("Choose Option", options)

	
	# --- Project Overview Section ---
	if section == "Project Overview":
		# Project Overview
		st.title("Project Overview")

		# Introduction Section
		st.subheader("1.1. Introduction")
		st.write("""
		A news outlet has engaged Team2 as data science consultants to develop classification models using Python 
		and integrate them into a web application using Streamlit. This project showcases the practical implementation 
		of machine learning methods in natural language processing, specifically for the automated categorization of 
		news articles into predefined categories.
		""")

		# Objectives of the Project
		st.subheader("1.2. Objectives of the Project")
		st.write("""
		The objective of this project is to design and implement a robust machine learning system capable of accurately 
		classifying news articles based on their content. The solution will involve comprehensive data preprocessing, 
		feature extraction using advanced techniques such as Bag-of-Words and N-grams, and the training and evaluation 
		of machine learning models.
		The project also aims to deploy a user-friendly web application using Streamlit, enabling users to input articles 
		and receive real-time category predictions. The automated system will reduce manual effort, minimize errors, 
		and improve the overall efficiency of news categorization for digital news platforms.
		""")

		# Data Source Section
		st.subheader("1.3. Data Source")
		st.write("The data can be accessed at: [GitHub Repository](https://github.com/DareSandtech/2407FTDS_Classification_Project)")

		# Problem Statement Section
		st.subheader("1.5. Problem Statement")
		st.write("""
		In today's digital era, news outlets face significant challenges in efficiently categorizing a vast array of 
		articles across diverse domains such as Business, Technology, Sports, Education, and Entertainment. 
		Manual classification processes are labor-intensive, time-consuming, and prone to errors, leading to inefficiencies 
		in organizing and retrieving content. This creates a pressing need for an automated solution that can accurately 
		and swiftly categorize news articles based on their content, improving operational efficiency and reducing human error.
		""")

		# Methodology Overview Section
		st.subheader("1.7. Methodology Overview")
		st.write("""
		1. **Data Uploading**: Upload the dataset provided and ensure it is properly loaded.
		2. **Data Preprocessing**: Clean and preprocess the text data by removing stopwords, punctuation, and special characters. 
		Perform tokenization, stemming, or lemmatization to standardize the text. Handle missing or inconsistent data.
		3. **Feature Extraction**: Transform the preprocessed text into numerical features using techniques like Bag-of-Words and N-grams.
		4. **Model Selection and Training**: Experiment with machine learning algorithms (e.g., Random Forest, Naive Bayes, Support Vector Machines, and Recurrent Neural Networks). 
		Train the models on the preprocessed and feature-extracted dataset.
		5. **Model Evaluation**: Evaluate model performance using accuracy, precision, recall, and F1-score. 
		Use cross-validation to ensure generalization to unseen data.
		6. **Model Optimization**: Fine-tune hyperparameters and experiment with feature extraction techniques to improve performance. 
		Address issues like overfitting or class imbalance.
		7. **Deployment**: Develop a user-friendly web application using Streamlit to allow users to input articles and receive predictions.
		8. **Testing and Validation**: Test the deployed system with real-world data to ensure it performs as expected.
		""")

	# --- Importing Packages Section ---
	if section == "Importing Packages":
		st.header("2. Importing Packages")
		
		# Code for importing packages (displayed as text)
		code = """
		import re
		import pandas as pd
		import numpy as np
		import matplotlib.pyplot as plt
		import seaborn as sns

		import nltk
		from nltk.stem import PorterStemmer
		from nltk.tokenize import word_tokenize
		from nltk import TreebankWordTokenizer
		from nltk.stem import WordNetLemmatizer
		from nltk.corpus import stopwords
		import unicodedata
		import string
		from sklearn.model_selection import st.session_state.train_st.session_state.test_split, cross_val_score
		from sklearn.preprocessing import StandardScaler
		from sklearn.linear_model import LinearRegression
		from sklearn.ensemble import RandomForestRegressor
		from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
		from sklearn.feature_extraction.text import TfidfVectorizer
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.metrics import classification_report
		from sklearn.feature_extraction.text import CountVectorizer
		from wordcloud import WordCloud
		from sklearn.preprocessing import LabelEncoder
		from sklearn.neighbors import KNeighborsClassifier
		from sklearn.metrics import accuracy_score, classification_report
		from sklearn.model_selection import GridSearchCV
		from IPython.display import display
		"""
		
		# Display the code as plain text
		st.code(code, language='python')

		# Add button to run the code
		run_button = st.button("Run")

		if run_button:
			try:
				st.success("Importing Packages completed successfully!")
			except Exception as e:
				st.error(f"Error executing code: {e}")
			
	# --- Loading Data Section ---
	if section == "Loading Data":
		st.header("3. Loading Data")
		st.write("The data used for this project is located in the test.csv and test.csv file. To better manipulate and analyse the data, it is loaded into a Pandas DataFrame using the Pandas function, .read_csv() and referred to as df.")
		
		# Code for loading the data
		code_loading = """
		# Load the data
		st.session_state.test = pd.read_csv(r'Data\\processed\\test.csv')
		st.session_state.train = pd.read_csv(r'Data\\processed\\train.csv')

		# Display the DataFrames
		st.subheader("Preview of the DataFrames")
		st.write("Train DataFrame:")
		st.dataframe(st.session_state.train)

		st.write("Test DataFrame:")
		st.dataframe(st.session_state.test)
		"""
		
		st.code(code_loading, language='python')

		# Add button to run the loading code
		load_button = st.button("Run")

		if "load_data" in st.session_state or load_button:
			try:
				# Load the data
				st.session_state.test = pd.read_csv('app/Data/processed/test.csv')
				st.session_state.train = pd.read_csv('app/Data/processed/test.csv')

				# Display the DataFrames
				st.subheader("Preview of the DataFrames")
				st.write("Train DataFrame:")
				st.dataframe(st.session_state.train)  

				st.write("Test DataFrame:")
				st.dataframe(st.session_state.test)  
				st.session_state.load_data = True

			except Exception as e:
				st.error(f"Error loading data: {e}")
				st.write("Please check the file paths or the data files.")

	# --- Initial Data Inspection Section ---
	if section == "Initial Data Inspection":
		st.header("4. Initial Data Inspection")
		st.write("The .shape attribute is used to retrieve the dimensions of the DataFrame, representing the number of rows and columns as a tuple.")
		st.write("The first dataset consists of 2000 observations (rows) and 5 features (columns). The second dataset consists of 5520 observations (rows) and 5 features (columns). The .info() method is used to generate a summary of the DataFrame's Features, including the column names, the count of null values and the datatype for each corresponding column.")
		# Show the code for initial data inspection
		code_initial_inspection = """
		# Display the shape of the datasets
		st.subheader("Shape of the Datasets")
		st.write("Train Data Shape:")
		st.write(f"Rows: {st.session_state.train.shape[0]} | Columns: {st.session_state.train.shape[1]}")

		st.write("Test Data Shape:")
		st.write(f"Rows: {st.session_state.test.shape[0]} | Columns: {st.session_state.test.shape[1]}")

		# Display basic statistics of the datasets
		st.subheader("Basic Statistics of the Datasets")
		st.write("Train Data Statistics:")
		st.write(st.session_state.train.describe(include='all').T)

		st.write("Test Data Statistics:")
		st.write(st.session_state.test.describe(include='all').T)
		"""
		
		# Display the code above the button
		st.code(code_initial_inspection, language='python')
		
		# Add button to trigger initial data inspection
		inspect_button = st.button("Run")

		if "data_inspection" in st.session_state or inspect_button:
			try:
				# Display the shape of the datasets
				st.subheader("Shape of the Datasets")
				st.write("Train Data Shape:")
				st.write(f"Rows: {st.session_state.train.shape[0]} | Columns: {st.session_state.train.shape[1]}")

				st.write("Test Data Shape:")
				st.write(f"Rows: {st.session_state.test.shape[0]} | Columns: {st.session_state.test.shape[1]}")

				# Display basic statistics of the datasets
				st.subheader("Basic Statistics of the Datasets")
				st.write("Train Data Statistics:")
				st.write(st.session_state.train.describe(include='all').T)

				st.write("Test Data Statistics:")
				st.write(st.session_state.test.describe(include='all').T)
				st.session_state.data_inspection = True

			except Exception as e:
				st.error(f"Error inspecting data: {e}")
				st.write("Please check the data files.")

	# --- Data Cleaning ---
	if section == "Data Cleaning":
		st.header("5. Data Cleaning")
		
		# --- Missing Values Check ---
		st.subheader("Check for Missing Values")
		st.write("""
		The `check_null_values` function serves as a utility to quickly identify and report any null values present in each column of a DataFrame.
		By iterating through each column and using the `.isnull()` and `.sum()` methods, it calculates the count of null values in each column.
		The function then prints out the count of null values alongside the corresponding column name, providing a clear overview of the null value distribution.
		""")

		# Code to display for checking missing values
		code_missing_values = """
		# Check for missing values in st.session_state.train and st.session_state.test data
		train_null = st.session_state.train.isnull().sum()
		test_null = st.session_state.test.isnull().sum()

		st.subheader("Missing Values in train Data:")
		st.write(train_null)

		st.subheader("Missing Values in test Data:")
		st.write(test_null)
		"""

		st.code(code_missing_values, language='python')

		# Button to run missing values check
		check_null_values_button = st.button("Run", key=1)

		if "check_null" in st.session_state or check_null_values_button:
			try:
				# Specify the columns to check for null values
				columns_to_check = ['headlines', 'description', 'content', 'url', 'category']

				# Check for missing values in the specified columns
				train_null = st.session_state.train[columns_to_check].isnull().sum()
				test_null = st.session_state.test[columns_to_check].isnull().sum()

				st.subheader("Missing Values in train Data:")
				st.write(train_null)

				st.subheader("Missing Values in test Data:")
				st.write(test_null)
				st.session_state.check_null = True

			except Exception as e:
				st.error(f"Error checking missing values: {e}")
				st.write("Please check the data files.")

		# --- Duplicates Check ---
		st.subheader("Check for Duplicates")
		st.write("""
		The `check_duplicated` function identifies any duplicate rows in the dataset using the `.duplicated()` method.
		This will help you identify if there are any rows that are identical across all columns.
		""")

		# Code to display for checking duplicates
		code_duplicates = """
		# Check for duplicates in train and test data
		train_duplicates = st.session_state.train.duplicated().sum()
		test_duplicates = st.session_state.test.duplicated().sum()

		st.subheader("Duplicates in train Data:")
		st.write(f"Number of duplicate rows: {train_duplicates}")

		st.subheader("Duplicates in test Data:")
		st.write(f"Number of duplicate rows: {test_duplicates}")
		"""

		st.code(code_duplicates, language='python')

		# Button to run duplicates check
		check_duplicates_button = st.button("Run", key=2)

		if "duplicates" in st.session_state or check_duplicates_button:
			try:
			# Specify the columns to check for null values
				columns_to_check = ['headlines', 'description', 'content', 'url', 'category']

				# Check for missing values in the specified columns
				train_duplicates = st.session_state.train[columns_to_check].duplicated().sum()
				test_duplicates = st.session_state.test[columns_to_check].duplicated().sum()

				st.subheader("Duplicates in train Data:")
				st.write(f"Number of duplicate rows: {train_duplicates}")

				st.subheader("Duplicates in test Data:")
				st.write(f"Number of duplicate rows: {test_duplicates}")
				st.session_state.duplicates = True

			except Exception as e:
				st.error(f"Error checking duplicates: {e}")
				st.write("Please check the data files.")

		st.header("6. Data Preprocessing")
		st.markdown("""In this section of the code, the following steps are performed to clean and preprocess the text data in the `train` and `test` datasets:

1. **Combining Text Columns**: 
   - The `headlines`, `description`, and `content` columns are combined into a single `text` column by joining their values. This ensures that all relevant information is available in one place for further cleaning.
  
2. **Lowercasing**: 
   - All text is converted to lowercase using the `.str.lower()` method. This ensures consistency by making the text case-insensitive during further processing.

3. **Removing Punctuation and Special Characters**: 
   - Punctuation marks and special characters are removed using regular expressions (`re.sub`). Only alphabetic characters and spaces are retained for a cleaner text.

4. **Tokenization**: 
   - The text is split into individual words (tokens) using the `word_tokenize` function. This allows for a more granular analysis of the text.

5. **Removing Stopwords**: 
   - Common stopwords (such as "the", "and", "is", etc.) are removed using the `stopwords.words('english')` set from the NLTK library. Stopwords do not provide much meaningful information and can be discarded.

6. **Lemmatization**: 
   - The words in the text are lemmatized using the `WordNetLemmatizer`. This step converts words to their base or dictionary form (e.g., "running" becomes "run"), helping to standardize the text.

7. **Final Text for Model Input**: 
   - The lemmatized words are joined back into full sentences using `' '.join(text)` to prepare the text for use in machine learning models.

At the end of this process, the `trainX` and `testX` variables contain the cleaned, tokenized, stopword-free, and lemmatized text, which can now be used for model training and evaluation.""")
		# Full code for preprocessing
		code_preprocessing = """
		# Combine 'headlines', 'description', 'content', and 'url' into one column 'text'
		st.session_state.train['text'] = st.session_state.train[['headlines', 'description', 'content', 'url']].agg(' '.join, axis=1)
		st.session_state.test['text'] = st.session_state.test[['headlines', 'description', 'content', 'url']].agg(' '.join, axis=1)

		# Convert all text to lowercase
		st.session_state.train['cleaned_text'] = st.session_state.train['text'].str.lower()
		st.session_state.test['cleaned_text'] = st.session_state.test['text'].str.lower()

		# Remove punctuations and special characters
		st.session_state.train['cleaned_text'] = st.session_state.train['cleaned_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
		st.session_state.test['cleaned_text'] = st.session_state.test['cleaned_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

		# Tokenizing the combined cleaned text
		st.session_state.train['tokenized_text'] = st.session_state.train['cleaned_text'].apply(word_tokenize)
		st.session_state.test['tokenized_text'] = st.session_state.test['cleaned_text'].apply(word_tokenize)

		# Remove stopwords from the tokenized text
		stopwords_list = set(stopwords.words('english'))
		st.session_state.train['removed_stopwords_text'] = st.session_state.train['tokenized_text'].apply(lambda text: [word for word in text if word not in stopwords_list])
		st.session_state.test['removed_stopwords_text'] = st.session_state.test['tokenized_text'].apply(lambda text: [word for word in text if word not in stopwords_list])

		# Initialize the lemmatizer
		lemmatizer = WordNetLemmatizer()

		# Lemmatize the train and test datasets
		st.session_state.train['lemmatized_text'] = st.session_state.train['removed_stopwords_text'].apply(lambda text: [lemmatizer.lemmatize(word) for word in text])
		st.session_state.test['lemmatized_text'] = st.session_state.test['removed_stopwords_text'].apply(lambda text: [lemmatizer.lemmatize(word) for word in text])

		# Display the cleaned data after lemmatization
		st.subheader("Data After Lemmatization")
		st.write("Train Data (Head):")
		st.dataframe(st.session_state.train.head())

		st.write("Test Data (Head):")
		st.dataframe(st.session_state.test.head())
		"""

		st.code(code_preprocessing, language='python')
        
		# Button to run preprocessing pipeline
		run_preprocessing_button = st.button("Run", key=3)

		if "preprocessing" in st.session_state :
			# Display the cleaned data after lemmatization
				st.subheader("Data After Processing")
				st.write("Train Data (Head):")
				st.dataframe(st.session_state.train.head())

				st.write("Test Data (Head):")
				st.dataframe(st.session_state.test.head())

		if run_preprocessing_button:
			try:
				# Combine 'headlines', 'description', 'content', and 'url' into one column 'text'
				st.session_state.train['text'] = st.session_state.train[['headlines', 'description', 'content', 'url']].agg(' '.join, axis=1)
				st.session_state.test['text'] = st.session_state.test[['headlines', 'description', 'content', 'url']].agg(' '.join, axis=1)

				# Convert all text to lowercase
				st.session_state.train['cleaned_text'] = st.session_state.train['text'].str.lower()
				st.session_state.test['cleaned_text'] = st.session_state.test['text'].str.lower()

				# Remove punctuations and special characters
				st.session_state.train['cleaned_text'] = st.session_state.train['cleaned_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
				st.session_state.test['cleaned_text'] = st.session_state.test['cleaned_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

				# Tokenizing the combined cleaned text
				st.session_state.train['tokenized_text'] = st.session_state.train['cleaned_text'].apply(word_tokenize)
				st.session_state.test['tokenized_text'] = st.session_state.test['cleaned_text'].apply(word_tokenize)

				# Remove stopwords from the tokenized text
				stopwords_list = set(stopwords.words('english'))
				st.session_state.train['removed_stopwords_text'] = st.session_state.train['tokenized_text'].apply(lambda text: [word for word in text if word not in stopwords_list])
				st.session_state.test['removed_stopwords_text'] = st.session_state.test['tokenized_text'].apply(lambda text: [word for word in text if word not in stopwords_list])

				# Initialize the lemmatizer
				lemmatizer = WordNetLemmatizer()

				# Lemmatize the train and test datasets
				st.session_state.train['lemmatized_text'] = st.session_state.train['removed_stopwords_text'].apply(lambda text: [lemmatizer.lemmatize(word) for word in text])
				st.session_state.test['lemmatized_text'] = st.session_state.test['removed_stopwords_text'].apply(lambda text: [lemmatizer.lemmatize(word) for word in text])

				# Display the cleaned data after lemmatization
				st.subheader("Data After Processing")
				st.write("Train Data (Head):")
				st.dataframe(st.session_state.train.head())

				st.write("Test Data (Head):")
				st.dataframe(st.session_state.test.head())
				st.session_state.preprocessing = True

			except Exception as e:
				st.error(f"Error during preprocessing: {e}")
				st.write("Please check the data or ensure necessary libraries are imported.")

	# --- EDA ---
	if section == "EDA": 
		st.header("1. Univariate Analysis")

		# Category Distribution in Train Data
		st.subheader("Category Distribution in Train Data")
		st.code("""
		fig, ax = plt.subplots(figsize=(10, 6))
		st.session_state.train['category'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
		ax.set_title("Article Distribution in Train Data")
		ax.set_xlabel("Category")
		ax.set_ylabel("Count")
		ax.tick_params(axis="x", rotation=45)
		st.pyplot(fig)
		""", language='python')

		if "train_bar" in st.session_state or st.button("Run - Category Distribution (Train)"):
			fig, ax = plt.subplots(figsize=(10, 6))
			st.session_state.train['category'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
			ax.set_title("Article Distribution in Train Data")
			ax.set_xlabel("Category")
			ax.set_ylabel("Count")
			ax.tick_params(axis="x", rotation=45)
			st.pyplot(fig)
			st.session_state.train_bar=True

		# Category Distribution in Test Data
		st.subheader("Category Distribution in Test Data")
		st.code("""
		fig, ax = plt.subplots(figsize=(10, 6))
		st.session_state.test['category'].value_counts().plot(kind='bar', color='lightcoral', ax=ax)
		ax.set_title("Article Distribution in Test Data")
		ax.set_xlabel("Category")
		ax.set_ylabel("Count")
		ax.tick_params(axis="x", rotation=45)
		st.pyplot(fig)
		""", language='python')

		if "test_bar" in st.session_state or st.button("Run - Category Distribution (Test)"):
			fig, ax = plt.subplots(figsize=(10, 6))
			st.session_state.test['category'].value_counts().plot(kind='bar', color='lightcoral', ax=ax)
			ax.set_title("Article Distribution in Test Data")
			ax.set_xlabel("Category")
			ax.set_ylabel("Count")
			ax.tick_params(axis="x", rotation=45)
			st.pyplot(fig)
			st.session_state.test_bar=True

		# --- Most Common Words ---
		st.header("3. Most Common Words in Lemmatized Text")

		st.code("""
# Define a function to get the top N most common words from a specific column (using lemmatized text)
def get_top_n_common_words(dataframe, column, n=5):
	# Join the list of tokens into a single string
	words = dataframe[column].apply(lambda x: ' '.join(x))  # Join list of tokens into a single string
	
	# Tokenize and count word frequencies using CountVectorizer
	vectorizer = CountVectorizer(stop_words='english')  # Removed max_features to get all words
	word_counts = vectorizer.fit_transform(words)
	
	# Sum the counts to get word frequencies
	word_freq = word_counts.sum(axis=0).A1
	
	# Create a list of words and their frequencies
	word_freq_dict = dict(zip(vectorizer.get_feature_names_out(), word_freq))
	
	# Sort words by frequency in descending order and get the top N words
	sorted_word_freq = sorted(word_freq_dict.items(), key=lambda item: item[1], reverse=True)
	
	# Get the top N words and their frequencies
	top_n_words = sorted_word_freq[:n]
	
	return top_n_words

# Columns to process (using lemmatized text column instead of original columns)
columns = ['lemmatized_text']  # Use the lemmatized text column

top_5_words_train = {}
top_5_words_test = {}

# For train dataframe
top_5_words_train['lemmatized_text'] = get_top_n_common_words(st.session_state.train, 'lemmatized_text')

# For test dataframe
top_5_words_test['lemmatized_text'] = get_top_n_common_words(st.session_state.test, 'lemmatized_text')

# Print the top 5 most common words for both train and test data
st.markdown("Top 5 common words in train dataframe (lemmatized text):")
for word, freq in top_5_words_train['lemmatized_text']:
	st.write(f"'{word}' with frequency {freq}")

st.markdown("\nTop 5 common words in test dataframe (lemmatized text):")
for word, freq in top_5_words_test['lemmatized_text']:
	st.write(f"'{word}' with frequency {freq}")
		""", language='python')

		if "most_common_words" in st.session_state or st.button("Run - Most Common Words"):
			# Define a function to get the top N most common words from a specific column (using lemmatized text)
			def get_top_n_common_words(dataframe, column, n=5):
				# Join the list of tokens into a single string
				words = dataframe[column].apply(lambda x: ' '.join(x))  # Join list of tokens into a single string
				
				# Tokenize and count word frequencies using CountVectorizer
				vectorizer = CountVectorizer(stop_words='english')  # Removed max_features to get all words
				word_counts = vectorizer.fit_transform(words)
				
				# Sum the counts to get word frequencies
				word_freq = word_counts.sum(axis=0).A1
				
				# Create a list of words and their frequencies
				word_freq_dict = dict(zip(vectorizer.get_feature_names_out(), word_freq))
				
				# Sort words by frequency in descending order and get the top N words
				sorted_word_freq = sorted(word_freq_dict.items(), key=lambda item: item[1], reverse=True)
				
				# Get the top N words and their frequencies
				top_n_words = sorted_word_freq[:n]
				
				return top_n_words

			# Columns to process (using lemmatized text column instead of original columns)
			columns = ['lemmatized_text']  # Use the lemmatized text column

			top_5_words_train = {}
			top_5_words_test = {}

			# For train dataframe
			top_5_words_train['lemmatized_text'] = get_top_n_common_words(st.session_state.train, 'lemmatized_text')

			# For test dataframe
			top_5_words_test['lemmatized_text'] = get_top_n_common_words(st.session_state.test, 'lemmatized_text')

			# Print the top 5 most common words for both train and test data
			st.markdown("Top 5 common words in train dataframe (lemmatized text):")
			for word, freq in top_5_words_train['lemmatized_text']:
				st.write(f"'{word}' with frequency {freq}")

			st.markdown("\nTop 5 common words in test dataframe (lemmatized text):")
			for word, freq in top_5_words_test['lemmatized_text']:
				st.write(f"'{word}' with frequency {freq}")

			st.session_state.most_common_words=True

		# --- Unique Words ---
		st.header("4. Unique Words in Combined Train & Test Data")

		st.code("""
# Function to get unique words (words that appear exactly once in the entire combined dataframe)
def get_unique_words_combined(train_df, test_df, column):
	# Combine train and test dataframes
	combined_df = pd.concat([train_df[column], test_df[column]], ignore_index=True)
	
	# Combine all lemmatized text from the selected columns into a single string per row
	combined_text = combined_df[column[0]].dropna().astype(str).str.cat(sep=' ')  # Combine lemmatized text column directly
	
	# Tokenize and count word frequencies using CountVectorizer
	vectorizer = CountVectorizer(stop_words='english') 
	word_counts = vectorizer.fit_transform([combined_text])
	
	# Sum the counts to get word frequencies
	word_freq = word_counts.sum(axis=0).A1
	
	# Create a list of words and their frequencies
	word_freq_dict = dict(zip(vectorizer.get_feature_names_out(), word_freq))
	
	# Find unique words (words that appear exactly once)
	unique_words = [word for word, freq in word_freq_dict.items() if freq == 1]
	
	return unique_words

# Column to analyze 
column_to_analyze = ['lemmatized_text']  

# Get unique words from both train and test dataframes combined
unique_words_combined = get_unique_words_combined(st.session_state.train, st.session_state.test, column_to_analyze)

st.markdown(f"**{len(unique_words_combined)}** unique words found")

st.markdown("Here are the first 20 unique words:")
st.write(unique_words_combined[:20])
		""", language='python')

		if "unique_words" in st.session_state or st.button("Run - Unique Words"):
			# Function to get unique words (words that appear exactly once in the entire combined dataframe)
			def get_unique_words_combined(train_df, test_df, column):
				# Combine train and test dataframes
				combined_df = pd.concat([train_df[column], test_df[column]], ignore_index=True)
				
				# Combine all lemmatized text from the selected columns into a single string per row
				combined_text = combined_df[column[0]].dropna().astype(str).str.cat(sep=' ')  # Combine lemmatized text column directly
				
				# Tokenize and count word frequencies using CountVectorizer
				vectorizer = CountVectorizer(stop_words='english') 
				word_counts = vectorizer.fit_transform([combined_text])
				
				# Sum the counts to get word frequencies
				word_freq = word_counts.sum(axis=0).A1
				
				# Create a list of words and their frequencies
				word_freq_dict = dict(zip(vectorizer.get_feature_names_out(), word_freq))
				
				# Find unique words (words that appear exactly once)
				unique_words = [word for word, freq in word_freq_dict.items() if freq == 1]
				
				return unique_words

			# Column to analyze 
			column_to_analyze = ['lemmatized_text']  

			# Get unique words from both train and test dataframes combined
			unique_words_combined = get_unique_words_combined(st.session_state.train, st.session_state.test, column_to_analyze)

			st.markdown(f"**{len(unique_words_combined)}** unique words found")

			st.markdown("Here are the first 20 unique words:")
			st.write(unique_words_combined[:20])

		st.header("Multivariate Analysis")

		# --- Word Cloud Visualization ---
		st.header("2. Word Cloud for Lemmatized Text Data")

		st.code("""
		fig, ax = plt.subplots(figsize=(10, 6))
		text = st.session_state.train['lemmatized_text'].dropna().astype(str).str.cat(sep=' ') + ' ' + st.session_state.test['lemmatized_text'].dropna().astype(str).str.cat(sep=' ')
		wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
		ax.imshow(wordcloud, interpolation='bilinear')
		ax.axis('off')
		ax.set_title("Word Cloud for Text Data")
		st.pyplot(fig)
		""", language='python')

		if "word_cloud" in st.session_state or st.button("Run - Word Cloud"):
			fig, ax = plt.subplots(figsize=(10, 6))
			text = st.session_state.train['lemmatized_text'].dropna().astype(str).str.cat(sep=' ') + ' ' + st.session_state.test['lemmatized_text'].dropna().astype(str).str.cat(sep=' ')
			wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
			ax.imshow(wordcloud, interpolation='bilinear')
			ax.axis('off')
			ax.set_title("Word Cloud for Lemmatized Text Data")
			st.pyplot(fig)
			st.session_state.word_cloud=True

	# Data Preparation Section
	if section == "Data Preparation":
		st.header("Data Preparation")
		st.markdown("""### Label Encoding, TF-IDF Vectorization, and Data Splitting

In this section, the following steps are performed:

1. **Label Encoding**:
   - The `LabelEncoder` is used to convert the categorical `category` labels into numerical form for both the training and test datasets. This ensures that the model can process the labels as numerical values.

2. **TF-IDF Vectorization**:
   - The `TfidfVectorizer` is applied to convert the `lemmatized_text` into a sparse matrix of TF-IDF features. This step transforms the text data into numerical format, using the top 3000 most important features (words).

3. **Labels (Target Variable)**:
   - The target variable, `category_encoded`, is assigned to `y_train` and `y_test`, corresponding to the encoded category labels for both training and testing datasets.

4. **Splitting the Training Data**:
   - The training dataset (`X_train`, `y_train`) is split into training and validation sets using an 80/20 split. The validation set will be used to evaluate model performance during training.

These steps prepare the data for model training by encoding the labels, vectorizing the text, and splitting the data into training and validation sets for evaluation.
""")


		st.code("""
		#  Label Encoding for Categories (Fit on Full Data)
		label_encoder = LabelEncoder()
		st.session_state.label_encoder =label_encoder
		label_encoder.fit(st.session_state.train['category'])  # Fit on train categories

		# Transform labels for training and test sets
		st.session_state.train['category_encoded'] = label_encoder.transform(st.session_state.train['category'])
		st.session_state.test['category_encoded'] = label_encoder.transform(st.session_state.test['category'])  # Apply same encoding

		# TF-IDF Vectorization (Fit on Train, Transform Both Train & Test)
		vectorizer = TfidfVectorizer(max_features=3000)
		X_train = vectorizer.fit_transform(st.session_state.train['lemmatized_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else ""))
		X_test = vectorizer.transform(st.session_state.test['lemmatized_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else ""))

		# Labels (Target Variable)
		y_train = st.session_state.train['category_encoded']
		y_test = st.session_state.test['category_encoded']  # Ensure labels match transformed test set

		# Split Train Set into Train and Validation
		X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

		# Save the prepared datasets to session state
		st.session_state.X_train = X_train
		st.session_state.X_test = X_test
		st.session_state.X_validation = X_validation
		st.session_state.y_train = y_train
		st.session_state.y_test = y_test
		st.session_state.y_validation = y_validation

				
		st.session_state.scaler = StandardScaler()
		# Scale the features using StandardScaler
		st.session_state.X_train_scaled = st.session_state.scaler.fit_transform(X_train.toarray())
		st.session_state.X_test_scaled = st.session_state.scaler.transform(X_test.toarray())
		st.session_state.X_validation_scaled = st.session_state.scaler.fit_transform(X_validation.toarray())
				
		st.success("Data Preparation Completed!")

		# Display information about the prepared data
		st.write(f"Training data shape: {X_train.shape}")
		st.write(f"Validation data shape: {X_validation.shape}")
		st.write(f"Test data shape: {X_test.shape}")
		st.write(f"Number of unique categories: {len(st.session_state.train['category_encoded'].unique())}")
		""", language="python")

		# Option to run the data preparation process
		if "preparation" in st.session_state or st.button("Run - Prepare Data"):
			# Assuming `st.session_state.train` and `st.session_state.test` are the datasets already loaded
			label_encoder = LabelEncoder()
			st.session_state.label_encoder =label_encoder
			label_encoder.fit(st.session_state.train['category'])  # Fit on train categories

			# Transform labels for training and test sets
			st.session_state.train['category_encoded'] = label_encoder.transform(st.session_state.train['category'])
			st.session_state.test['category_encoded'] = label_encoder.transform(st.session_state.test['category'])  # Apply same encoding

			# TF-IDF Vectorization (Fit on Train, Transform Both Train & Test)
			vectorizer = TfidfVectorizer(max_features=3000)
			X_train = vectorizer.fit_transform(st.session_state.train['lemmatized_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else ""))
			X_test = vectorizer.transform(st.session_state.test['lemmatized_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else ""))

			# Labels (Target Variable)
			y_train = st.session_state.train['category_encoded']
			y_test = st.session_state.test['category_encoded']  # Ensure labels match transformed test set

			# Split Train Set into Train and Validation
			X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

			# Save the prepared datasets to session state
			st.session_state.X_train = X_train
			st.session_state.X_test = X_test
			st.session_state.X_validation = X_validation
			st.session_state.y_train = y_train
			st.session_state.y_test = y_test
			st.session_state.y_validation = y_validation

			st.session_state.scaler = StandardScaler()
			# Scale the features using StandardScaler
			st.session_state.X_train_scaled = st.session_state.scaler.fit_transform(X_train.toarray())
			st.session_state.X_test_scaled = st.session_state.scaler.transform(X_test.toarray())
			st.session_state.X_validation_scaled = st.session_state.scaler.fit_transform(X_validation.toarray())

			st.success("Data Preparation Completed!")
			st.session_state.preparation = True
			# Display information about the prepared data
			st.write(f"Training data shape: {X_train.shape}")
			st.write(f"Validation data shape: {X_validation.shape}")
			st.write(f"Test data shape: {X_test.shape}")
			st.write(f"Number of unique categories: {len(st.session_state.train['category_encoded'].unique())}")

	# Model Building and Tuning Section
	if section == "Model Building, Training, Tuning and Testing":
		st.header("Model Building Random Forest")

		# Display code before execution
		st.code("""
#build model
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Fit model
start_time = time.time()
rf_model.fit(st.session_state.X_train, st.session_state.y_train)
end_time = time.time()

# Save the model in session_state
st.session_state.rf_model = rf_model

# Calculate and show training time
training_time_rf = end_time - start_time
st.session_state.training_time_rf=training_time_rf

#predict
y_pred_rf = rf_model.predict(st.session_state.X_test)

# Save best model to session_state
st.session_state.rf_model = rf_model

# Evaluate the tuned model
accuracy_rf_tuned = accuracy_score(st.session_state.y_test, y_pred_rf)
st.write(f"Random Forest Accuracy: {accuracy_rf_tuned:.2f}")

# Plot confusion matrix for the tuned model
plot_confusion_matrix(st.session_state.y_test, y_pred_rf, st.session_state.train['category'].unique(), title="Random Forest Confusion Matrix")

# Classification report for the tuned model
results_rf_tuned = classification_report(st.session_state.y_test, y_pred_rf, target_names=st.session_state.train['category'].unique(), zero_division=1)
st.code(results_rf_tuned)
	""", language='python')

		# run Random Forest Model
		if "rf" in st.session_state or st.button("Run - Random Forest Model"):
			#build model
			rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
			
			# Fit model
			start_time = time.time()
			rf_model.fit(st.session_state.X_train, st.session_state.y_train)
			end_time = time.time()
			
			# Save the model in session_state
			st.session_state.rf_model = rf_model
			
			# Calculate and show training time
			training_time_rf = end_time - start_time
			st.session_state.training_time_rf=training_time_rf
			
			# Predict
			y_pred_rf = rf_model.predict(st.session_state.X_test)
			
			# Save best model to session_state
			st.session_state.rf_model = rf_model
			
			# Evaluate the tuned model
			accuracy_rf= accuracy_score(st.session_state.y_test, y_pred_rf)
			st.write(f"Random Forest Accuracy: {accuracy_rf:.2f}")

			# Plot confusion matrix for the tuned model
			plot_confusion_matrix(st.session_state.y_test, y_pred_rf, st.session_state.train['category'].unique(), title="Random Forest Confusion Matrix")
			
			# Classification report for the tuned model
			results_rf = classification_report(st.session_state.y_test, y_pred_rf, target_names=st.session_state.train['category'].unique(), zero_division=1)
			st.code(results_rf)

			st.success(f"Model training completed in {training_time_rf:.2f} seconds!")

			st.session_state.rf = True

		st.header("Model Building KNN")

		# Display code before execution
		st.code("""# Initialize and Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
start_time = time.time()
knn.fit(st.session_state.X_train_scaled, st.session_state.y_train)
end_time = time.time()

# Make Predictions and Evaluate
y_pred_knn = knn.predict(st.session_state.X_validation_scaled)
accuracy_knn = accuracy_score(st.session_state.y_validation, y_pred_knn)
st.write(f"KNN Accuracy: {accuracy_knn:.2f}")

# Calculate training time
training_time_knn = end_time - start_time
st.session_state.training_time_knn=training_time_knn

# Plot Confusion Matrix
plot_confusion_matrix(st.session_state.y_validation, y_pred_knn, st.session_state.label_encoder.classes_, title="Tuned KNN Confusion Matrix")

# Classification Report for Model
results_knn = classification_report(st.session_state.y_validation, y_pred_knn, target_names=st.session_state.label_encoder.classes_, zero_division=1)
st.code(results_knn)

# Save the best model to session state
st.session_state.knn = knn
	""", language='python')

		# run KNN
		if "KNN" in st.session_state or st.button("Run - KNN Model"):
			# Initialize and Train KNN Classifier
			knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
			start_time = time.time()
			knn.fit(st.session_state.X_train_scaled, st.session_state.y_train)
			end_time = time.time()

			# Make Predictions and Evaluate
			y_pred_knn = knn.predict(st.session_state.X_validation_scaled)
			accuracy_knn = accuracy_score(st.session_state.y_validation, y_pred_knn)
			st.write(f"KNN Accuracy: {accuracy_knn:.2f}")

			# Calculate training time
			training_time_knn = end_time - start_time
			st.session_state.training_time_knn=training_time_knn

			# Plot Confusion Matrix
			plot_confusion_matrix(st.session_state.y_validation, y_pred_knn, st.session_state.label_encoder.classes_, title="Tuned KNN Confusion Matrix")
			
			# Classification Report for Model
			results_knn = classification_report(st.session_state.y_validation, y_pred_knn, target_names=st.session_state.label_encoder.classes_, zero_division=1)
			st.code(results_knn)

			# Save the best model to session state
			st.session_state.knn = knn

			st.success(f"KNN Model Training Completed in {training_time_knn:.2f} seconds!")

			st.session_state.KNN = True

		st.header("Naive Bayes Model Building")

		# Display code before execution
		st.code("""# Ensure X_train and y_train are already available in session_state
X_train = st.session_state.X_train
y_train = st.session_state.y_train
X_validation = st.session_state.X_validation
y_validation = st.session_state.y_validation

# Compute Class Weights to Handle Class Imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Initialize & Train Multinomial Naïve Bayes
nb = MultinomialNB()
start_time = time.time()
nb.fit(X_train, y_train, sample_weight=sample_weights)
end_time = time.time()
# Save the model in session_state
st.session_state.nb_model = nb

# Make Predictions and Evaluate
y_pred_nb = nb.predict(X_validation)

# Print Accuracy, Confusion Matrix and Classification Report
accuracy_nb = accuracy_score(y_validation, y_pred_nb)
st.write(f"Naive Bayes Accuracy: {accuracy_nb:.2f}")

plot_confusion_matrix(y_validation, y_pred_nb, st.session_state.train['category'].unique(), title="Naive Bayes Confusion Matrix")

results_nb = classification_report(y_validation, y_pred_nb, target_names=st.session_state.train['category'].unique(), zero_division=1)
st.write("Naive Bayes Classification Report:")
st.code(results_nb)

# Calculate training duration
training_time_nb = end_time - start_time
st.session_state.training_time_nb=training_time_nb
	""", language="python")

		# run Naive Bayes 
		if "nb" in st.session_state or st.button("Run - Naive Bayes Model"):
			# Ensure X_train and y_train are already available in session_state
			X_train = st.session_state.X_train
			y_train = st.session_state.y_train
			X_validation = st.session_state.X_validation
			y_validation = st.session_state.y_validation
			
			# Compute Class Weights to Handle Class Imbalance
			sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

			# Initialize & Train Multinomial Naïve Bayes
			nb = MultinomialNB()
			start_time = time.time()
			nb.fit(X_train, y_train, sample_weight=sample_weights)
			end_time = time.time()
			# Save the model in session_state
			st.session_state.nb_model = nb
			
			# Make Predictions and Evaluate
			y_pred_nb = nb.predict(X_validation)

			# Print Accuracy, Confusion Matrix and Classification Report
			accuracy_nb = accuracy_score(y_validation, y_pred_nb)
			st.write(f"Naive Bayes Accuracy: {accuracy_nb:.2f}")

			plot_confusion_matrix(y_validation, y_pred_nb, st.session_state.train['category'].unique(), title="Naive Bayes Confusion Matrix")

			results_nb = classification_report(y_validation, y_pred_nb, target_names=st.session_state.train['category'].unique(), zero_division=1)
			st.write("Naive Bayes Classification Report:")
			st.code(results_nb)
			
			# Calculate training duration
			training_time_nb = end_time - start_time
			st.session_state.training_time_nb=training_time_nb

			st.success(f"Model training completed in {training_time_nb:.2f} seconds!")

			st.session_state.nb = True

		st.header("Neural network (MLP Classifier) Model Building")

		# Display code before execution
		st.code("""# Get session state variables
X_train = st.session_state.X_train
y_train = st.session_state.y_train
X_validation = st.session_state.X_validation
y_validation = st.session_state.y_validation
X_train_scaled = st.session_state.X_train_scaled
X_validation_scaled = st.session_state.X_validation_scaled

# Initialize and Train Neural Network (MLP Classifier)
mlp = MLPClassifier(
hidden_layer_sizes=(100,),  # 1 hidden layer with 100 neurons
max_iter=500,  # Increased for better convergence
activation='relu',  # Good for deep learning models
solver='adam',  # Works well with large datasets
early_stopping=True,  # Stops training if validation score stops improving
random_state=42,
verbose=False  # Prints progress during training
)

# Start the timer
start_time = time.time()
mlp.fit(X_train_scaled, y_train)  # Train the model
# End the timer
end_time = time.time()

# Calculate  training duration
training_time_nn = end_time - start_time
st.session_state.training_time_nn=training_time_nn

# Make Predictions and Evaluate Performance
y_pred_mlp = mlp.predict(X_validation_scaled)

# Print Accuracy and Classification Report
accuracy = accuracy_score(y_validation, y_pred_mlp)
st.write(f"MLP Accuracy: {accuracy:.2f}")

plot_confusion_matrix(y_validation, y_pred_mlp, label_encoder.classes_, title="Neural network Confusion Matrix")

results_nn = classification_report(y_validation, y_pred_mlp, target_names=label_encoder.classes_, zero_division=1)
st.write("Neural Network Classification Report:")
st.code(results_nn)
	""", language='python')

		# run Neural network
		if "nn" in st.session_state or st.button("Run - Neural network Model"):
			# Get session state variables
			X_train = st.session_state.X_train
			y_train = st.session_state.y_train
			X_validation = st.session_state.X_validation
			y_validation = st.session_state.y_validation
			X_train_scaled = st.session_state.X_train_scaled
			X_validation_scaled = st.session_state.X_validation_scaled

			# Initialize and Train Neural Network (MLP Classifier)
			mlp = MLPClassifier(
				hidden_layer_sizes=(100,),  # 1 hidden layer with 100 neurons
				max_iter=500,  # Increased for better convergence
				activation='relu',  # Good for deep learning models
				solver='adam',  # Works well with large datasets
				early_stopping=True,  # Stops training if validation score stops improving
				random_state=42,
				verbose=False  # Prints progress during training
			)

			# Start the timer
			start_time = time.time()
			mlp.fit(X_train_scaled, y_train)  # Train the model
			# End the timer
			end_time = time.time()

			# Calculate  training duration
			training_time_nn = end_time - start_time
			st.session_state.training_time_nn=training_time_nn

			# Make Predictions and Evaluate Performance
			y_pred_mlp = mlp.predict(X_validation_scaled)

			# Print Accuracy and Classification Report
			accuracy = accuracy_score(y_validation, y_pred_mlp)
			st.write(f"MLP Accuracy: {accuracy:.2f}")

			plot_confusion_matrix(y_validation, y_pred_mlp, st.session_state.label_encoder.classes_, title="Neural network Confusion Matrix")
			
			results_nn = classification_report(y_validation, y_pred_mlp, target_names=st.session_state.label_encoder.classes_, zero_division=1)
			st.write("Neural Network Classification Report:")
			st.code(results_nn)

			st.session_state.mlp=mlp
			st.session_state.nn = True
		
		st.header("Logistic Regression Model Building")

		# Display code before execution
		code = ''' # Get session state variables
X_train = st.session_state.X_train
y_train = st.session_state.y_train
X_validation = st.session_state.X_validation
y_validation = st.session_state.y_validation

# Initialize Logistic Regression and fit
log_reg = LogisticRegression()
# Start the timer
start_time = time.time()
log_reg.fit(X_train, y_train)  # Ensure X_train is the TF-IDF transformed dataset
# End the timer
end_time = time.time()

# Calculate and print training duration
training_time_lr = end_time - start_time
st.session_state.training_time_lr=training_time_lr

y_pred_lr = log_reg.predict(X_validation)

# Evaluate performance
accuracy = accuracy_score(y_validation, y_pred_lr)
st.write(f"Validation Accuracy: {accuracy:.2f}")

plot_confusion_matrix(y_validation, y_pred_lr, label_encoder.classes_, title="Tuned Random Forest Confusion Matrix")

results_lr = classification_report(y_validation, y_pred_lr, target_names=label_encoder.classes_, zero_division=1)
st.write("Logistic Regression Classification Report:")
st.code(results_lr)

st.session_state.log_reg = log_reg

	'''
		st.code(code, language="python")

		# run Logistic Regression
		if "lg" in st.session_state or st.button("Run - Logistic Regression Model"):
			# Get session state variables
			X_train = st.session_state.X_train
			y_train = st.session_state.y_train
			X_validation = st.session_state.X_validation
			y_validation = st.session_state.y_validation

			# Initialize Logistic Regression and fit
			log_reg = LogisticRegression()
			# Start the timer
			start_time = time.time()
			log_reg.fit(X_train, y_train)  # Ensure X_train is the TF-IDF transformed dataset
			# End the timer
			end_time = time.time()

			# Calculate and print training duration
			training_time_lr = end_time - start_time
			st.session_state.training_time_lr=training_time_lr

			y_pred_lr = log_reg.predict(X_validation)

			# Evaluate performance
			accuracy = accuracy_score(y_validation, y_pred_lr)
			st.write(f"Validation Accuracy: {accuracy:.2f}")

			plot_confusion_matrix(y_validation, y_pred_lr, st.session_state.label_encoder.classes_, title="Tuned Random Forest Confusion Matrix")

			results_lr = classification_report(y_validation, y_pred_lr, target_names=st.session_state.label_encoder.classes_, zero_division=1)
			st.write("Logistic Regression Classification Report:")
			st.code(results_lr)

			st.session_state.log_reg = log_reg

			st.success(f"Model training completed in {training_time_lr:.2f} seconds!")
			st.session_state.lg = True

		st.header("SVM model Building")

		# Display code before execution
		code_test = '''# Get session state variables
X_train = st.session_state.X_train
y_train = st.session_state.y_train
X_validation = st.session_state.X_validation
y_validation = st.session_state.y_validation

# **Train Initial SVM Model **
svm_model = SVC(kernel='rbf', gamma='scale', C=1.0, decision_function_shape='ovr')
start_time = time.time()
svm_model.fit(X_train, y_train)
end_time = time.time()
#pedict
y_pred_svm = svm_model.predict(X_validation)

# Calculate and print training duration
training_time_svm = end_time - start_time
st.session_state.training_time_svm=training_time_svm

accuracy_svm = accuracy_score(y_validation, y_pred_svm)
st.write(f"Validation Accuracy: {accuracy_svm:.2f}")

plot_confusion_matrix(y_validation, y_pred_svm, label_encoder.classes_, title="SVM Confusion Matrix")

# **Classification Report**
results_svm = classification_report(y_validation, y_pred_svm, target_names=label_encoder.classes_, zero_division=1)
st.write("SVM Classification Report:")
st.code(results_svm)
	'''
		st.code(code_test, language="python")
	
		# run SVM model
		if "SVM" in st.session_state or st.button("Run - SVM model"):
			# Get session state variables
			X_train = st.session_state.X_train
			y_train = st.session_state.y_train
			X_validation = st.session_state.X_validation
			y_validation = st.session_state.y_validation

			# **Train Initial SVM Model **
			svm_model = SVC(kernel='rbf', gamma='scale', C=1.0, decision_function_shape='ovr')
			start_time = time.time()
			svm_model.fit(X_train, y_train)
			end_time = time.time()
			#pedict
			y_pred_svm = svm_model.predict(X_validation)

			# Calculate and print training duration
			training_time_svm = end_time - start_time
			st.session_state.training_time_svm=training_time_svm

			accuracy_svm = accuracy_score(y_validation, y_pred_svm)
			st.write(f"Validation Accuracy: {accuracy_svm:.2f}")

			plot_confusion_matrix(y_validation, y_pred_svm, st.session_state.label_encoder.classes_, title="SVM Confusion Matrix")

			# **Classification Report**
			results_svm = classification_report(y_validation, y_pred_svm, target_names=st.session_state.label_encoder.classes_, zero_division=1)
			st.write("SVM Classification Report:")
			st.code(results_svm)

			st.session_state.svm_model = svm_model

			st.success(f"Model training completed in {training_time_svm:.2f} seconds!")
			st.session_state.SVM = True

		st.header("Base Model Performance Evaluation")

		st.subheader("Training time")

		# Display the code for Model Training Time Comparison
		code_training_time_comparison = '''
		# Define Model Names and Their Training Times
		models = ["Random Forest", "KNN", "Naïve Bayes", "Neural Network", "Logistic Regression", "SVM"]
		training_times = [
			st.session_state.training_time_rf,
			st.session_state.training_time_knn,
			st.session_state.training_time_nb,
			st.session_state.training_time_nn,
			st.session_state.training_time_lr,
			st.session_state.training_time_svm
		]

		# Fixing Scaling Issue (If Values are Too Large)
		max_time = max(training_times)
		if max_time > 100:  # Convert seconds to minutes if necessary
			training_times = [t / 60 for t in training_times]
			time_unit = "Minutes"
		else:
			time_unit = "Seconds"

		# Create a Bar Plot
		plt.figure(figsize=(10, 6))
		sns.barplot(x=training_times, y=models, palette="coolwarm")

		# Formatting and Labels
		plt.xlabel(f"Training Time ({time_unit})")
		plt.ylabel("Model")
		plt.title("Comparison of Model Training Times")
		plt.xlim(0, max(training_times) * 1.1)  # Set x-axis dynamically
		plt.grid(axis="x", linestyle="--", alpha=0.7)

		# Display the Training Time Values on Bars
		for index, value in enumerate(training_times):
			plt.text(value + (max(training_times) * 0.01), index, f"{value:.2f}", va='center', fontsize=12)

		# Show the Plot
		st.pyplot(plt)
		'''
		
		st.code(code_training_time_comparison, language="python")

		if "training_times1" in st.session_state or st.button("Run - Model Training Times"):
			# Define Model Names and Their Training Times
			st.session_state.models = ["Random Forest", "KNN", "Naïve Bayes", "Neural Network", "Logistic Regression", "SVM"]
			training_times = [
				st.session_state.training_time_rf,
				st.session_state.training_time_knn,
				st.session_state.training_time_nb,
				st.session_state.training_time_nn,
				st.session_state.training_time_lr,
				st.session_state.training_time_svm
			]

			# Fixing Scaling Issue (If Values are Too Large)
			max_time = max(training_times)
			if max_time > 100:  # Convert seconds to minutes if necessary
				training_times = [t / 60 for t in training_times]
				time_unit = "Minutes"
			else:
				time_unit = "Seconds"

			# Create a Bar Plot
			plt.figure(figsize=(10, 6))
			sns.barplot(x=training_times, y=st.session_state.models, palette="coolwarm")

			# Formatting and Labels
			plt.xlabel(f"Training Time ({time_unit})")
			plt.ylabel("Model")
			plt.title("Comparison of Model Training Times")
			plt.xlim(0, max(training_times) * 1.1)  # Set x-axis dynamically
			plt.grid(axis="x", linestyle="--", alpha=0.7)

			# Display the Training Time Values on Bars
			for index, value in enumerate(training_times):
				plt.text(value + (max(training_times) * 0.01), index, f"{value:.2f}", va='center', fontsize=12)

			# Show the Plot
			st.pyplot(plt)
			st.session_state.training_times1 = True

		st.subheader("Cross validation")

		# Display code before execution
		code_test_nn = ''' models = {
'Random Forest': st.session_state.rf_model,
'K Nearest Neighbors': st.session_state.knn,
'Naive Bayes': st.session_state.nb_model,
'Neural Networks': st.session_state.mlp,
'Logistic Regression': st.session_state.log_reg,
'Support Vector Machine': st.session_state.svm_model
}

# Initialize an empty list to store cross-validation results
cv = []

# Perform cross-validation for each model
for name, model in models.items():
st.write(f"### {name}")  

# Perform k-fold cross-validation with k=10
scores = cross_val_score(model, st.session_state.X_train, st.session_state.y_train, cv=10)

# Calculate the mean and standard deviation of the cross-validation scores
mean_score = scores.mean()
std_dev = scores.std()

st.write(f"Accuracy: {mean_score:0.2f} (+/- {std_dev:0.4f})")  # Display results

# Append the model name, mean cross-validation score, and standard deviation to the 'cv' list
cv.append([name, mean_score, std_dev])

# Convert the cross-validation results to a DataFrame
cv_df = pd.DataFrame(cv, columns=['Model', 'CV_Mean', 'CV_Std_Dev'])

# Display the DataFrame of cross-validation results
st.write("### Cross-validation Results", cv_df)

# Plot the results 
fig, ax = plt.subplots(figsize=(10, 6))
cv_df.plot(x='Model', y='CV_Mean', yerr='CV_Std_Dev', kind='bar', ax=ax, ylim=[0.95, 0.99])

# Display the plot
st.pyplot(fig)
		'''

		st.code(code_test_nn, language="python")

		# Option to test the model
		if "validation1" in st.session_state or st.button("Run - Cross validation"):
			models = {
				'Random Forest': st.session_state.rf_model,
				'K Nearest Neighbors': st.session_state.knn,
				'Naive Bayes': st.session_state.nb_model,
				'Neural Networks': st.session_state.mlp,
				'Logistic Regression': st.session_state.log_reg,
				'Support Vector Machine': st.session_state.svm_model
			}

			# Initialize an empty list to store cross-validation results
			cv = []

			# Perform cross-validation for each model
			for name, model in models.items():
				st.write(f"### {name}")  
				
				# Perform k-fold cross-validation with k=10
				scores = cross_val_score(model, st.session_state.X_train, st.session_state.y_train, cv=10)
				
				# Calculate the mean and standard deviation of the cross-validation scores
				mean_score = scores.mean()
				std_dev = scores.std()
				
				st.write(f"Accuracy: {mean_score:0.2f} (+/- {std_dev:0.4f})")  # Display results
				
				# Append the model name, mean cross-validation score, and standard deviation to the 'cv' list
				cv.append([name, mean_score, std_dev])

			# Convert the cross-validation results to a DataFrame
			cv_df = pd.DataFrame(cv, columns=['Model', 'CV_Mean', 'CV_Std_Dev'])

			# Display the DataFrame of cross-validation results
			st.write("### Cross-validation Results", cv_df)

			# Plot the results 
			fig, ax = plt.subplots(figsize=(10, 6))
			cv_df.plot(x='Model', y='CV_Mean', yerr='CV_Std_Dev', kind='bar', ax=ax, ylim=[0.95, 0.99])

			# Display the plot
			st.pyplot(fig)
			st.session_state.validation1 = True

		st.header("Tuning of Top 3 models using Grid Search")
		st.subheader("Logistic regression")

		# Display the code before execution
		code_test_log_reg = ''' 
param_grid = {
'C': [0.01, 0.1, 1, 10, 100],   # Regularization strength
'penalty': ['l2', 'l1'],  # 'l1' only works with 'liblinear' and 'saga'
'solver': ['liblinear', 'saga'],  # 'saga' supports all penalties
'max_iter': [200, 500, 1000]  # Increase iterations to prevent convergence issues
}

mlflow.set_experiment("Logistic Regression")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run():
# Initialize Logistic Regression (without fitting yet)
log_reg = LogisticRegression()

# Perform GridSearchCV
grid_log = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Start the timer
start_time = time.time()
grid_log.fit(st.session_state.X_train, st.session_state.y_train)  
# End the timer
end_time = time.time()

# Calculate and print training duration
training_time_log = end_time - start_time

# Use the best model from GridSearchCV
best_log = grid_log.best_estimator_
y_pred_best_log = best_log.predict(st.session_state.X_validation)

accuracy_best_log = accuracy_score(st.session_state.y_validation, y_pred_best_log)
classification_best_log = classification_report(st.session_state.y_validation, y_pred_best_log, target_names=st.session_state.label_encoder.classes_)

# Log metrics
mlflow.log_metric('accuracy_best_log', accuracy_best_log)
mlflow.log_text(classification_best_log, 'classification_best_log.txt')

# Log hyperparameters
mlflow.log_params(grid_log.best_params_)

# Log model
signature = infer_signature(st.session_state.X_train, best_log.predict(st.session_state.X_train))
mlflow.sklearn.log_model(best_log, "Best Logistic Regression", signature=signature)

with open("Best_Logistic_Regression.pkl", "wb") as file:
pickle.dump(best_log, file)

# best parameters and performance
st.write(f"Best_log_Parameters: {grid_log.best_params_}")
st.write(f"Best log Accuracy: {grid_log.best_score_:.2f}")

# Visualize Classification Report after tuning
st.write(f'Best Logistic Regression classification: {classification_best_log}')
		'''

		# Display the code in Streamlit
		st.code(code_test_log_reg, language="python")

		# Option to test the model
		if "lrt" in st.session_state or st.button("Run - Logistic Regression Grid Search"):
			mlflow.set_experiment("Logistic Regression1")
			mlflow.set_tracking_uri("http://127.0.0.1:5000")

			with mlflow.start_run():
				# Initialize Logistic Regression (without fitting yet)
				log_reg = LogisticRegression()

				param_grid = {
					'C': [0.01, 0.1, 1, 10, 100],   # Regularization strength
					'penalty': ['l2', 'l1'],  # 'l1' only works with 'liblinear' and 'saga'
					'solver': ['liblinear', 'saga'],  # 'saga' supports all penalties
					'max_iter': [200, 500, 1000]  # Increase iterations to prevent convergence issues
				}
				# Perform GridSearchCV
				grid_log = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

				# Start the timer
				start_time = time.time()
				grid_log.fit(st.session_state.X_train, st.session_state.y_train)  
				# End the timer
				end_time = time.time()

				# Calculate and print training duration
				training_time_log = end_time - start_time
                
				# Use the best model from GridSearchCV
				best_log = grid_log.best_estimator_
				y_pred_best_log = best_log.predict(st.session_state.X_validation)

				accuracy_best_log = accuracy_score(st.session_state.y_validation, y_pred_best_log)
				classification_best_log = classification_report(st.session_state.y_validation, y_pred_best_log, target_names=st.session_state.label_encoder.classes_)

				# Log metrics
				mlflow.log_metric('accuracy_best_log', accuracy_best_log)
				mlflow.log_text(classification_best_log, 'classification_best_log.txt')

				# Log hyperparameters
				mlflow.log_params(grid_log.best_params_)

				# Log model
				signature = infer_signature(st.session_state.X_train, best_log.predict(st.session_state.X_train))
				mlflow.sklearn.log_model(best_log, "Best Logistic Regression", signature=signature)

				with open("Best_Logistic_Regression.pkl", "wb") as file:
					pickle.dump(best_log, file)

				# best parameters and performance
				st.write(f"Best_log_Parameters: {grid_log.best_params_}")
				st.write(f"Best log Accuracy: {grid_log.best_score_:.2f}")

				# Visualize Classification Report after tuning
				st.write(f'Best Logistic Regression classification: {classification_best_log}')
			
			st.session_state.lrt = True

		st.subheader("Testing Best Logistic Regression ")
        # Display the code before execution
		code_test_log_reg = ''' 
# Prediction
y_test_pred_log = best_log.predict(X_test)

# Evaluate performance on the test set
accuracy_log_test = accuracy_score(y_test, y_test_pred_log)

# Display Accuracy
st.write(f"**Test Accuracy:** {accuracy_log_test:.2f}")

# Display classification report
results_test_lr = classification_report(y_test, y_test_pred_log, target_names=label_encoder.classes_, zero_division=1)
st.write("**Final Test Classification Report:**")
st.text(results_test_lr)

# Plot Confusion Matrix for the Test Data
st.write("**Confusion Matrix:**")
plot_confusion_matrix(y_test, y_test_pred_log, label_encoder.classes_, title="Final Test Confusion Matrix")
		'''

		# Display the code in Streamlit
		st.code(code_test_log_reg, language="python")

		# Option to test the model
		if "lrtt" in st.session_state or st.button("Run - Logistic Regression Test"):
			# Prediction
			y_test_pred_log = best_log.predict(X_test)
			st.session_state.y_test_pred_log = y_test_pred_log
	
			# Evaluate performance on the test set
			accuracy_log_test = accuracy_score(y_test, y_test_pred_log)
			st.session_state.accuracy_log_test = accuracy_log_test
			# Display Accuracy
			st.write(f"**Test Accuracy:** {accuracy_log_test:.2f}")

			# Display classification report
			results_test_lr = classification_report(y_test, y_test_pred_log, target_names=label_encoder.classes_, zero_division=1)
			st.write("**Final Test Classification Report:**")
			st.text(results_test_lr)

			# Plot Confusion Matrix for the Test Data
			st.write("**Confusion Matrix:**")
			plot_confusion_matrix(y_test, y_test_pred_log, label_encoder.classes_, title="Final Test Confusion Matrix")
			st.session_state.lrtt = True

		# Display Streamlit Header
		st.subheader("Support Vector Machine (SVM)")

		# Display the code before execution
		code_svm = ''' 
		param_grid = {
			'C': [0.1, 1, 10, 100],
			'gamma': [1, 0.1, 0.01, 0.001],
			'kernel': ['rbf', 'poly', 'sigmoid']
		}

		mlflow.set_experiment("Support_Vector_Machine")
		mlflow.set_tracking_uri("http://127.0.0.1:5000")

		with mlflow.start_run():

			# Initialize and fit the Grid Search
			grid_svm = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
			start_time = time.time()
			grid_svm.fit(X_train, y_train) 
			end_time = time.time()

			# Calculate and print training duration
			training_time_svm = end_time - start_time

			best_svm = grid_svm.best_estimator_
			y_pred_best_svm = best_svm.predict(X_validation)  

			# Accuracy and Classification Report
			accuracy_best_svm = accuracy_score(y_validation, y_pred_best_svm)
			classification_best_svm = classification_report(y_validation, y_pred_best_svm, target_names=label_encoder.classes_)

			# Log metrics
			mlflow.log_metric('accuracy_best_svm', accuracy_best_svm)
			mlflow.log_text(classification_best_svm, 'classification_best_svm.txt')

			# Log hyperparameters
			mlflow.log_params(grid_svm.best_params_)

			# Log model
			signature = infer_signature(X_train, best_svm.predict(X_train))
			mlflow.sklearn.log_model(best_svm, "Best Support Vector Machine", signature=signature)

			with open("Best_Support_Vector_Machine.pkl", "wb") as file:
				pickle.dump(best_svm, file)

			# Print the best parameters and estimator
			print("Best svm parameters: ", grid_svm.best_params_)
			print(f"Best svm Accuracy: , {grid_svm.best_score_:.2f}")

			# Visualize Classification Report after tuning
			print(f'Best Support Vector Machine classification: {classification_best_svm}')
		'''

		# Display code in Streamlit
		st.code(code_svm, language="python")

		# Option to run the model training
		if "svm_trained" in st.session_state or st.button("Run - Support Vector Machine Grid Search"):
			mlflow.set_experiment("Support_Vector_Machine")
			mlflow.set_tracking_uri("http://127.0.0.1:5000")

			with mlflow.start_run():
				# Initialize and perform GridSearchCV
				grid_svm = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
				start_time = time.time()
				grid_svm.fit(st.session_state.X_train, st.session_state.y_train) 
				end_time = time.time()

				# Calculate and print training duration
				training_time_svm = end_time - start_time
				st.session_state.training_time_svm = training_time_svm  # Store training time

				# Save best model
				best_svm = grid_svm.best_estimator_
				y_pred_best_svm = best_svm.predict(st.session_state.X_validation)

				# Accuracy and Classification Report
				accuracy_best_svm = accuracy_score(st.session_state.y_validation, y_pred_best_svm)
				classification_best_svm = classification_report(st.session_state.y_validation, y_pred_best_svm, target_names=st.session_state.label_encoder.classes_)

				# Log metrics to MLflow
				mlflow.log_metric('accuracy_best_svm', accuracy_best_svm)
				mlflow.log_text(classification_best_svm, 'classification_best_svm.txt')

				# Log hyperparameters
				mlflow.log_params(grid_svm.best_params_)

				# Log the model
				signature = infer_signature(st.session_state.X_train, best_svm.predict(st.session_state.X_train))
				mlflow.sklearn.log_model(best_svm, "Best Support Vector Machine", signature=signature)

				# Save the model with pickle
				with open("Best_Support_Vector_Machine.pkl", "wb") as file:
					pickle.dump(best_svm, file)

				# Store results in session state
				st.session_state.best_svm = best_svm
				st.session_state.y_pred_best_svm = y_pred_best_svm
				st.session_state.accuracy_best_svm = accuracy_best_svm
				st.session_state.classification_best_svm = classification_best_svm

				# Display the best parameters and performance
				st.write(f"Best SVM Parameters: {grid_svm.best_params_}")
				st.write(f"Best SVM Accuracy: {grid_svm.best_score_:.2f}")
				st.write(f'Best SVM Classification Report: {classification_best_svm}')
			
			st.session_state.svm_trained = True
		
			st.subheader("Testing the Best Support Vector Machine")

			# Display code for testing the model
			code_test_svm = ''' 
		# Predict on the Test Set using the Best Tuned Model
		y_test_pred_svm = best_svm.predict(X_test)

		# **Evaluate on the Test Set**
		accuracy_svm_test = accuracy_score(y_test, y_test_pred_svm)   
		print(f"Final Tuned SVM Test Accuracy: {accuracy_svm_test:.2f}")

		# **Classification Report for Test Set**
		results_svm_test = classification_report(y_test, y_test_pred_svm, target_names=label_encoder.classes_, zero_division=1)
		print("\nFinal Tuned SVM Classification Report:")
		print(results_svm_test)

		# **Confusion Matrix for Test Set**
		plot_confusion_matrix(y_test, y_test_pred_svm, label_encoder.classes_, title="Final Tuned SVM Test Confusion Matrix")
		'''
			st.code(code_test_svm, language="python")

		# Display the code before execution
		code_svm = ''' 
# Prediction on Test Set using the trained model
y_test_pred_svm = st.session_state.best_svm.predict(st.session_state.X_test)
st.session_state.y_test_pred_svm = y_test_pred_svm

# Evaluate on the Test Set
accuracy_svm_test = accuracy_score(st.session_state.y_test, y_test_pred_svm)
st.session_state.accuracy_svm_test = accuracy_svm_test

# Display Accuracy
st.write(f"**Test Accuracy:** {accuracy_svm_test:.2f}")

# Display classification report
results_svm_test = classification_report(st.session_state.y_test, y_test_pred_svm, target_names=st.session_state.label_encoder.classes_, zero_division=1)
st.write("**Final Test Classification Report:**")
st.text(results_svm_test)

# Plot Confusion Matrix for Test Set
st.write("**Confusion Matrix:**")
plot_confusion_matrix(st.session_state.y_test, y_test_pred_svm, st.session_state.label_encoder.classes_, title="Final Tuned SVM Test Confusion Matrix")

		'''

		# Display code in Streamlit
		st.code(code_svm, language="python")

		if "svm_tested" in st.session_state or st.button("Run - SVM Test"):
			# Prediction on Test Set using the trained model
			y_test_pred_svm = st.session_state.best_svm.predict(st.session_state.X_test)
			st.session_state.y_test_pred_svm = y_test_pred_svm

			# Evaluate on the Test Set
			accuracy_svm_test = accuracy_score(st.session_state.y_test, y_test_pred_svm)
			st.session_state.accuracy_svm_test = accuracy_svm_test

			# Display Accuracy
			st.write(f"**Test Accuracy:** {accuracy_svm_test:.2f}")

			# Display classification report
			results_svm_test = classification_report(st.session_state.y_test, y_test_pred_svm, target_names=st.session_state.label_encoder.classes_, zero_division=1)
			st.write("**Final Test Classification Report:**")
			st.text(results_svm_test)

			# Plot Confusion Matrix for Test Set
			st.write("**Confusion Matrix:**")
			plot_confusion_matrix(st.session_state.y_test, y_test_pred_svm, st.session_state.label_encoder.classes_, title="Final Tuned SVM Test Confusion Matrix")

			st.session_state.svm_tested = True
		
		# Display Streamlit Header
		st.subheader("Naive Bayes Model (with Hyperparameter Tuning)")

		# Display the code before execution
		code_nb = ''' 
# Compute Class Weights to Handle Class Imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Hyperparameter Tuning for Naive Bayes using GridSearchCV
param_grid_nb = {'alpha': [0.01, 0.1, 1, 10]}  # Smoothing parameter tuning
grid_search_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# MLFlow experiment setup
mlflow.set_experiment("Naive_Bayes_Model")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Start MLflow run for Naive Bayes
with mlflow.start_run():
	# Start timer for Naive Bayes training
	start_time = time.time()
	grid_search_nb.fit(X_train, y_train, sample_weight=sample_weights)
	end_time = time.time()

	# Calculate and log training time
	training_time_nb = end_time - start_time
	mlflow.log_metric('training_time_nb', training_time_nb)

	# Get best model and predictions
	best_nb_model = grid_search_nb.best_estimator_
	y_pred_nb_tuned = best_nb_model.predict(X_validation)

	# Log metrics and results
	accuracy_nb_tuned = accuracy_score(y_validation, y_pred_nb_tuned)
	mlflow.log_metric('accuracy_nb_tuned', accuracy_nb_tuned)

	# Classification report and confusion matrix
	classification_nb_tuned = classification_report(y_validation, y_pred_nb_tuned, target_names=label_encoder.classes_, zero_division=1)
	mlflow.log_text(classification_nb_tuned, 'classification_nb_tuned.txt')

	# Log the hyperparameters of the best model
	mlflow.log_params(grid_search_nb.best_params_)

	# Log the Naive Bayes model
	signature = infer_signature(X_train, best_nb_model.predict(X_train))
	mlflow.sklearn.log_model(best_nb_model, "Best Naive Bayes Model", signature=signature)

	# Save the best Naive Bayes model as a .pkl file
	with open("Best_Naive_Bayes_Model.pkl", "wb") as file:
		pickle.dump(best_nb_model, file)

	# Print the best parameters and accuracy
	print(f"Best Naive Bayes Parameters: {grid_search_nb.best_params_}")
	print(f"Tuned Naive Bayes Accuracy: {accuracy_nb_tuned:.2f}")

	# Visualize Classification Report after tuning
	print(f"Tuned Naive Bayes classification report: {classification_nb_tuned}")
	plot_confusion_matrix(y_validation, y_pred_nb_tuned, label_encoder.classes_, title="Confusion Matrix (Tuned Naive Bayes)")
		'''
		# Display code in Streamlit
		st.code(code_nb, language="python")

		# Option to run the model training
		if "naive_bayes_trained" in st.session_state or st.button("Run - Naive Bayes Grid Search"):
			# Set up MLFlow experiment
			mlflow.set_experiment("Naive_Bayes_Model")
			mlflow.set_tracking_uri("http://127.0.0.1:5000")

			# Start the MLFlow run
			with mlflow.start_run():
				# Compute class weights to handle imbalance
				sample_weights = compute_sample_weight(class_weight='balanced', y=st.session_state.y_train)

				# Hyperparameter tuning using GridSearchCV
				param_grid_nb = {'alpha': [0.01, 0.1, 1, 10]}
				grid_search_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

				# Training with time logging
				start_time = time.time()
				grid_search_nb.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=sample_weights)
				end_time = time.time()

				# Log training time
				training_time_nb = end_time - start_time
				st.session_state.training_time_nb = training_time_nb
				mlflow.log_metric('training_time_nb', training_time_nb)

				# Best model and predictions
				best_nb_model = grid_search_nb.best_estimator_
				y_pred_nb_tuned = best_nb_model.predict(st.session_state.X_validation)
				
				# Store results in session_state for later use
				st.session_state.best_nb_model = best_nb_model
				st.session_state.y_pred_nb_tuned = y_pred_nb_tuned

				# Log metrics and classification report
				accuracy_nb_tuned = accuracy_score(st.session_state.y_validation, y_pred_nb_tuned)
				st.session_state.accuracy_nb_tuned = accuracy_nb_tuned
				mlflow.log_metric('accuracy_nb_tuned', accuracy_nb_tuned)

				classification_nb_tuned = classification_report(st.session_state.y_validation, y_pred_nb_tuned, target_names=st.session_state.label_encoder.classes_, zero_division=1)
				st.session_state.classification_nb_tuned = classification_nb_tuned
				mlflow.log_text(classification_nb_tuned, 'classification_nb_tuned.txt')

				# Log best hyperparameters and model
				mlflow.log_params(grid_search_nb.best_params_)
				signature = infer_signature(st.session_state.X_train, best_nb_model.predict(st.session_state.X_train))
				mlflow.sklearn.log_model(best_nb_model, "Best Naive Bayes Model", signature=signature)

				# Save model as a .pkl file
				with open("Best_Naive_Bayes_Model.pkl", "wb") as file:
					pickle.dump(best_nb_model, file)

				# Display results in Streamlit
				st.write(f"Best Naive Bayes Parameters: {grid_search_nb.best_params_}")
				st.write(f"Tuned Naive Bayes Accuracy: {accuracy_nb_tuned:.2f}")
				st.write(f"Tuned Naive Bayes classification report: \n{classification_nb_tuned}")

				st.session_state.naive_bayes_trained = True

		st.subheader("Testing the Best Naive Bayes Model")

		# Display code for testing the model
		code_test_nb = ''' 
# Predict on the Test Set using the Trained Naive Bayes
y_pred_test_nb = best_nb_model.predict(X_test)

# Evaluate on the Test Set
accuracy_nb_test = accuracy_score(y_test, y_pred_test_nb) 
print(f"Final Naive Bayes Test Accuracy: {accuracy_nb_test:.2f}")

# Classification Report for Test Set
results_nb_test = classification_report(y_test, y_pred_test_nb, target_names=label_encoder.classes_, zero_division=1)
print("\\nFinal Naive Bayes Test Classification Report:")
print(results_nb_test)

# Confusion Matrix for Test Set
plot_confusion_matrix(y_test, y_pred_test_nb, label_encoder.classes_, title="Final Tuned Naive Bayes Test Confusion Matrix")
	'''
		st.code(code_test_nb, language="python")

		if "naive_bayes_tested" in st.session_state or st.button("Run - Naive Bayes Test"):
			# Prediction on Test Set using the trained model
			y_pred_test_nb = st.session_state.best_nb_model.predict(st.session_state.X_test)
			st.session_state.y_pred_test_nb = y_pred_test_nb

			# Evaluate on the Test Set
			accuracy_nb_test = accuracy_score(st.session_state.y_test, y_pred_test_nb)
			st.session_state.accuracy_nb_test = accuracy_nb_test

			# Display Accuracy
			st.write(f"**Test Accuracy:** {accuracy_nb_test:.2f}")

			# Display classification report
			results_nb_test = classification_report(st.session_state.y_test, y_pred_test_nb, target_names=st.session_state.label_encoder.classes_, zero_division=1)
			st.write("**Final Test Classification Report:**")
			st.text(results_nb_test)

			# Plot Confusion Matrix for Test Set
			st.write("**Confusion Matrix:**")
			plot_confusion_matrix(st.session_state.y_test, y_pred_test_nb, st.session_state.label_encoder.classes_, title="Final Tuned Naive Bayes Test Confusion Matrix")

			st.session_state.naive_bayes_tested = True
		
		st.header("Metrics to evaluate top 3 models")
		st.subheader("Comparing Accuracy of 3 best performing models")
		
		code_test_nb = ''' 
accuracies = [
    accuracy_log_test,  # Logistic Regression Test Accuracy
    accuracy_svm_test,  # SVM Test Accuracy
    accuracy_nb_test    # Naive Bayes Test Accuracy
]

# Model names for the bar plot
models = ['Logistic Regression', 'Support Vector Machine', 'Naive Bayes']

# Create a Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=accuracies, y=models, palette="coolwarm")

# Formatting and Labels
plt.xlabel("Test Accuracy")
plt.ylabel("Model")
plt.title("Comparison of Model Test Accuracies")
plt.xlim(0.5, 1.0)  # Adjust x-axis for better visibility
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Display the Accuracy Values on Bars
for index, value in enumerate(accuracies):
    plt.text(value + 0.01, index, f"{value:.2f}", va='center', fontsize=12)

# Show the Plot using Streamlit
st.pyplot(plt)
	'''
		st.code(code_test_nb, language="python")

		if "naive_bayes_tested" in st.session_state or "svm_tested" in st.session_state or "lrtt" in st.session_state or st.button("Run - Compare Accuracy"):
			accuracies = [
				st.session_state.accuracy_log_test,  # Logistic Regression Test Accuracy
				st.session_state.accuracy_svm_test,  # SVM Test Accuracy
				st.session_state.accuracy_nb_test    # Naive Bayes Test Accuracy
			]

			# Model names for the bar plot
			models = ['Logistic Regression', 'Support Vector Machine', 'Naive Bayes']

			# Create a Bar Plot
			plt.figure(figsize=(10, 6))
			sns.barplot(x=accuracies, y=models, palette="coolwarm")

			# Formatting and Labels
			plt.xlabel("Test Accuracy")
			plt.ylabel("Model")
			plt.title("Comparison of Model Test Accuracies")
			plt.xlim(0.5, 1.0)  # Adjust x-axis for better visibility
			plt.grid(axis="x", linestyle="--", alpha=0.7)

			# Display the Accuracy Values on Bars
			for index, value in enumerate(accuracies):
				plt.text(value + 0.01, index, f"{value:.2f}", va='center', fontsize=12)

			# Show the Plot using Streamlit
			st.pyplot(plt)

		st.subheader("Comparing Precision of 3 best performing models")

		code_test_nb = ''' 
# Define Model Names and Their Predictions
models = ["Logistic Regression", "SVM", "Naive Bayes"]
y_preds = [y_test_pred_log, y_test_pred_svm, y_pred_test_nb]

# Compute Precision Scores for Each Model
precision_scores = [precision_score(y_test, y_pred, average='weighted') for y_pred in y_preds]

# Create a Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=precision_scores, palette="coolwarm")

# Formatting & Labels
plt.xlabel("Models")
plt.ylabel("Weighted Average Precision Score")
plt.title("Precision Scores Across Different Models")
plt.ylim(0.5, 1.0)  # Adjust range dynamically if needed
plt.xticks(rotation=45)  # Rotate labels for better visibility
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Display Precision Values on Bars
for index, value in enumerate(precision_scores):
    plt.text(index, value + 0.005, f"{value:.2f}", ha='center', fontsize=12)

# Show the Plot using Streamlit
st.pyplot(plt)
	'''
		st.code(code_test_nb, language="python")

		if "naive_bayes_tested" in st.session_state or "svm_tested" in st.session_state or "lrtt" in st.session_state or st.button("Run - Compare Precision"):
			# Define Model Names and Their Predictions (Ensure these variables are defined before use)
			models = ["Logistic Regression", "SVM", "Naive Bayes"]
			y_preds = [st.session_state.y_test_pred_log, st.session_state.y_test_pred_svm, st.session_state.y_pred_test_nb]

			# Compute Precision Scores for Each Model
			precision_scores = [precision_score(y_test, y_pred, average='weighted') for y_pred in y_preds]

			# Create a Bar Plot
			plt.figure(figsize=(10, 6))
			sns.barplot(x=models, y=precision_scores, palette="coolwarm")

			# Formatting & Labels
			plt.xlabel("Models")
			plt.ylabel("Weighted Average Precision Score")
			plt.title("Precision Scores Across Different Models")
			plt.ylim(0.5, 1.0)  # Adjust range dynamically if needed
			plt.xticks(rotation=45)  # Rotate labels for better visibility
			plt.grid(axis="y", linestyle="--", alpha=0.7)

			# Display Precision Values on Bars
			for index, value in enumerate(precision_scores):
				plt.text(index, value + 0.005, f"{value:.2f}", ha='center', fontsize=12)

			# Show the Plot using Streamlit
			st.pyplot(plt)

	# Section for prediction
	if section == "Prediction":
		st.info("Prediction with ML Models")

		# User input for text
		news_text = st.text_area("Enter Text", "Type Here")

		# Dropdown to select model
		model_choice = st.selectbox("Choose a model", ["SVM", "Naive Bayes", "Logistic Regression"])

		# Initialize session state for 'test' and 'model_paths' if they don't exist
		if 'test' not in st.session_state:
			st.session_state.test = {}

		if 'model_paths' not in st.session_state.test:
			st.session_state.test['model_paths'] = {}

		# If "Classify" button is clicked
		if st.button("Classify"):
			# Update model paths in session state
			st.session_state.test['model_paths'] = {
				"SVM": 'app/Streamlit/Best_Support_Vector_Machine.pkl',
				"Naive Bayes": 'app/Streamlit/Best_Naive_Bayes_Model.pkl',
				"Logistic Regression": 'app/Streamlit/Best_Logistic_Regression.pkl',
			}
			st.session_state.test['vectorizer_path'] = 'app/Streamlit//tfidfvect.pkl'

			# Check if vectorizer and selected model exist
			if os.path.exists(st.session_state.test['vectorizer_path']) and os.path.exists(st.session_state.test['model_paths'][model_choice]):
				# Load the vectorizer and selected model
				tfidf_vectorizer = joblib.load(open(st.session_state.test['vectorizer_path'], "rb"))
				model = joblib.load(open(st.session_state.test['model_paths'][model_choice], "rb"))

				# Transform the user input using the vectorizer
				vect_text = tfidf_vectorizer.transform([news_text]).toarray()

				# Predict using the selected model
				prediction = model.predict(vect_text)

				# Mapping prediction to readable format (assuming categories are numerical)
				label_map = {0: "business", 1: "education", 2: "entertainment", 3: "sports", 4: "technology"}  # Example categories
				predicted_label = label_map.get(prediction[0], "Unknown Category")

				# Display the result
				st.success(f"Text Categorized as: {predicted_label}")
			else:
				st.error("Model or Vectorizer not found. Please ensure the files are in place.")


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
