import numpy as np
import pandas as pd
import time
from scipy.spatial import cKDTree



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

print(len(dictionary))

# convert words to vectors
def word_to_character_vectors(word):
    character_vectors = [0] * 26
    for char in word:
        if char.isalpha():
            character_vectors[ord(char.lower()) - ord('a')] = 1
    return character_vectors


character_vectors_list = [word_to_character_vectors(word) for word in dictionary]

if not character_vectors_list:
    print("No valid words found in the provided word list.")
else:

    character_vectors_2d = np.array(character_vectors_list).reshape(len(character_vectors_list), -1)

    kdtree = cKDTree(character_vectors_2d)

    def find_nearest_word(word, target_length):
        character_vectors = np.array(word_to_character_vectors(word)).reshape(1, -1)  # Reshape to 2D
        #_, nearest_indices = kdtree.query(character_vectors, k=1)

        # Filter words by length
        candidate_words = [w for w in dictionary if len(w) == target_length]

        if not candidate_words:
            return word  # Return the original word if no candidates with the target length are found

        # Find the nearest word among candidates
        candidate_matrix = np.array([word_to_character_vectors(w) for w in candidate_words]).reshape(len(candidate_words), -1)
        _, nearest_index = cKDTree(candidate_matrix).query(character_vectors, k=1)

        return candidate_words[nearest_index[0]]

    def correct_text(input_text):
        words = input_text.split()
        corrected_words = []

        for word in words:
            if word not in dictionary:
                corrected_word = find_nearest_word(word, len(word))
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
