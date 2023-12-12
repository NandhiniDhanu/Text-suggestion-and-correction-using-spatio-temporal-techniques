import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer



test_data=pd.read_csv("test_dataset.csv")
test_data=np.array(test_data)


#from nltk.corpus import words
#file_path = "words_alpha.txt"
#file_path="words.txt"
file_path="10000_Words.txt"
# Read the words from the file
with open(file_path, 'r') as file:
    word_list = [line.strip() for line in file]

# Convert the list of words to a NumPy array
dictionary = np.array(word_list)


# Get lowercase dictionary
#dictionary = [word.lower() for word in words.words()]

# Maximum number of characters in a word
max_word_length = 15

# Convert words to binary vectors using TF-IDF
#tfidf_vectorizer = TfidfVectorizer(analyzer='char', lowercase=True, binary=True)
#tfidf_matrix = tfidf_vectorizer.fit_transform(dictionary)
#print(tfidf_matrix[1000])
# ...

# Convert words to binary vectors using TF-IDF
tfidf_vectorizer = TfidfVectorizer(analyzer='char', lowercase=True, binary=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(dictionary)

# Check the shape of tfidf_matrix
print("Shape of tfidf_matrix:", tfidf_matrix.shape)

if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
    print("No valid words found in the provided word list.")
else:
  
    kdtree = cKDTree(tfidf_matrix.toarray())

    def find_nearest_word(word, target_length):
        word_vector = tfidf_vectorizer.transform([word]).toarray()
        _, nearest_indices = kdtree.query(word_vector, k=1)

        # Filter words by length
        candidate_words = [w for w in dictionary if len(w) == target_length]

        if not candidate_words:
            return word  # Return the original word if no candidates with the target length are found

        # Find the nearest word among candidates
        candidate_matrix = tfidf_vectorizer.transform(candidate_words).toarray()
        _, nearest_index = cKDTree(candidate_matrix).query(word_vector, k=1)

        return candidate_words[nearest_index[0]]

    def correct_text(input_text):
        words = input_text.split()
        corrected_words = []

        for word in words:
            if word.lower() not in dictionary:
                corrected_word = find_nearest_word(word.lower(), len(word))
            else:
                corrected_word = word

            corrected_words.append(corrected_word)

        corrected_input_text = " ".join(corrected_words)
        return corrected_input_text

    print("Enter text: ")
    
    input_text = input()
    start_time = time.time()
    corrected_text = correct_text(input_text)

    print("Input_text:", input_text)
    print("Corrected text:", corrected_text)
    end_time = time.time()
    total_time = end_time - start_time
    print('Execution Time: ', total_time)

    correct_count = 0
    total_count = len(test_data)

    for input_text, expected_output in test_data:
        corrected_text = correct_text(input_text)
        if corrected_text == expected_output:
            correct_count += 1

    accuracy = correct_count / total_count
    print(f"Accuracy: {accuracy * 100:.2f}%")
