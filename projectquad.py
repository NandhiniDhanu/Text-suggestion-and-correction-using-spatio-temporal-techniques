import numpy as np
import time
import pandas as pd

test_data = pd.read_csv("test_dataset.csv")
test_data = np.array(test_data)

#file_path = "words_alpha.txt"
#file_path="words.txt"
file_path="10000_Words.txt"

# Read the words from the file
with open(file_path, 'r') as file:
    word_list = [line.strip() for line in file]

# Convert the list of words to a NumPy array
dictionary = np.array(word_list)

def word_to_character_vectors(word):
    character_vectors = [0] * 26
    for char in word:
        if char.isalpha():
            character_vectors[ord(char.lower()) - ord('a')] = 1
    return character_vectors

character_vectors_list = [word_to_character_vectors(word) for word in dictionary]

# Simple QuadTree implementation
class QuadTree:
    def __init__(self, boundary, capacity, depth=0, max_depth=10):
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.is_divided = False
        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None
        self.depth = depth
        self.max_depth = max_depth

    def insert(self, point, index):
        if not self.in_boundary(point):
            return False

        if len(self.points) < self.capacity or self.depth >= self.max_depth:
            self.points.append((point, index))
            return True
        else:
            if not self.is_divided:
                self.subdivide()

            if self.nw.insert(point, index):
                return True
            elif self.ne.insert(point, index):
                return True
            elif self.sw.insert(point, index):
                return True
            elif self.se.insert(point, index):
                return True
            else:
                return False

    def subdivide(self):
        x, y = self.boundary[0]
        w = self.boundary[1]
        h = self.boundary[2]

        ne_boundary = [(x + w / 2, y), w / 2, h / 2]
        nw_boundary = [(x, y), w / 2, h / 2]
        se_boundary = [(x + w / 2, y + h / 2), w / 2, h / 2]
        sw_boundary = [(x, y + h / 2), w / 2, h / 2]

        self.nw = QuadTree(ne_boundary, self.capacity, self.depth + 1, self.max_depth)
        self.ne = QuadTree(nw_boundary, self.capacity, self.depth + 1, self.max_depth)
        self.sw = QuadTree(sw_boundary, self.capacity, self.depth + 1, self.max_depth)
        self.se = QuadTree(se_boundary, self.capacity, self.depth + 1, self.max_depth)
        self.is_divided = True

    def in_boundary(self, point):
        x, y = point[0], point[1]
        x0, y0 = self.boundary[0]
        w = self.boundary[1]
        h = self.boundary[2]
        return (x >= x0 and x < x0 + w and y >= y0 and y < y0 + h)

# Create the QuadTree with a maximum depth limit
boundary = [[0, 0], 1, 1]  # Example boundary
quadtree = QuadTree(boundary, 4, max_depth=10)

for idx, char_vector in enumerate(character_vectors_list):
    point = char_vector
    quadtree.insert(point, idx)

def find_nearest_word(word, target_length):
    character_vector = np.array(word_to_character_vectors(word))
    nearest_word = None
    min_distance = float('inf')

    def recursive_search(node):
        nonlocal nearest_word, min_distance
        if node.points:
            for point, index in node.points:
                distance = np.linalg.norm(character_vector - np.array(point))
                if distance < min_distance and len(dictionary[index]) == target_length:
                    min_distance = distance
                    nearest_word = dictionary[index]

        if node.is_divided:
            recursive_search(node.nw)
            recursive_search(node.ne)
            recursive_search(node.sw)
            recursive_search(node.se)

    recursive_search(quadtree)
    return nearest_word

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
