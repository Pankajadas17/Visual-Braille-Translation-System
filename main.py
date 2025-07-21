import numpy as np
import cv2
import os
import re
from gtts import gTTS
from dotenv import load_dotenv
import requests
from transformers import pipeline

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

braille_dict = {
    "100000": "A", "100001": "B", "110000": "C", "111000": "D", "101000": "E",
    "110001": "F", "111001": "G", "101001": "H", "010001": "I", "011001": "J",
    "100010": "K", "100011": "L", "110010": "M", "111010": "N", "101010": "O",
    "110011": "P", "111011": "Q", "101011": "R", "010011": "S", "011011": "T",
    "100110": "U", "100111": "V", "011101": "W", "110110": "X", "111110": "Y", "101110": "Z"
}

def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([A-Z])", r" \1", text).replace("  ", " ")
    return text.capitalize()

def enhance_with_llm(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
    return response.json()['choices'][0]['message']['content'].strip()

# Optional NER & Sentiment
sentiment_pipeline = pipeline("sentiment-analysis")
ner_pipeline = pipeline("ner", grouped_entities=True)

def analyze_sentiment_ner(text):
    sentiment = sentiment_pipeline(text)
    entities = ner_pipeline(text)
    return sentiment, entities

def calculate_distance(pnt1, pnt2):
    return ((pnt1[1] - pnt2[1]) ** 2 + (pnt1[0] - pnt2[0]) ** 2) ** 0.5

def clockwise_traverse(c_label):
    b_string = ""
    weight = 0
    x_co, y_co = int(centers_list[c_label][0]), int(centers_list[c_label][1])
    for dx, dy in [(0, 0), (0, shortest_x), (shortest_x, 0), (shortest_x, 0), (0, -shortest_x), (-shortest_x, 0)]:
        x_co += dx
        y_co += dy
        label = label_matrix[x_co][y_co]
        if label == 0:
            b_string += "0"
        else:
            b_string += "1"
            dots_traversed.append(label)
            weight += 1
    return b_string, weight

def anticlockwise_traverse(a_label):
    b_string = ""
    weight = 0
    x_co, y_co = int(centers_list[a_label][0]), int(centers_list[a_label][1])
    for dx, dy in [(0, 0), (0, -shortest_x), (shortest_x, 0), (shortest_x, 0), (0, shortest_x), (-shortest_x, 0)]:
        x_co += dx
        y_co += dy
        label = label_matrix[x_co][y_co]
        if label == 0:
            b_string += "0"
        else:
            b_string += "1"
            dots_traversed.append(label)
            weight += 1
    b_string = b_string[2:] + b_string[:2]
    return b_string[::-1], weight

# Load image
im = cv2.imread("Braille.png", 0)
im_padded = np.pad(im, 1, "constant", constant_values=255)
im_padded = (im_padded // 128) * 255
size_rows, size_columns = im_padded.shape

black, white = 0, 255
label_matrix = np.zeros((size_rows, size_columns))
label = 0
list_label = {}
arrange_array = []

# 8-connectivity
for i in range(1, size_rows - 1):
    for j in range(1, size_columns - 1):
        if im_padded[i][j] == black:
            if all(label_matrix[i + dx][j + dy] == 0 for dx, dy in [(-1, 0), (0, -1), (-1, -1), (-1, 1)]):
                label += 1
                label_matrix[i][j] = label
                list_label[label] = label
            else:
                for dx, dy in [(-1, 0), (0, -1), (-1, -1), (-1, 1)]:
                    if label_matrix[i + dx][j + dy] != 0:
                        arrange_array.append(label_matrix[i + dx][j + dy])
                min_val = min(arrange_array)
                label_matrix[i][j] = min_val
                for x in arrange_array:
                    if x in list_label:
                        list_label[x] = list_label[min_val]
                arrange_array = []

for i in range(1, size_rows - 1):
    for j in range(1, size_columns - 1):
        if label_matrix[i][j] != 0:
            label_matrix[i][j] = list_label[label_matrix[i][j]]

unique_digits = np.unique(label_matrix)
unique_digits = np.delete(unique_digits, [0])
count = len(unique_digits)

# Center point of dots
centers_list = {}
for i in range(1, len(list_label) + 1):
    if list_label[i] == i:
        index = np.where(label_matrix == i)
        x_mid = np.ceil(np.mean(index[0]))
        y_mid = np.ceil(np.mean(index[1]))
        centers_list[i] = (x_mid, y_mid)

# Distance between dots
p1, p2 = centers_list[unique_digits[0]], centers_list[unique_digits[1]]
shortest_x = int(calculate_distance(p1, p2))
for i in range(1, count - 1):
    p1, p2 = centers_list[unique_digits[i]], centers_list[unique_digits[i + 1]]
    dist = calculate_distance(p1, p2)
    if dist < shortest_x:
        shortest_x = int(dist)

final_string = ""
dots_traversed = []
space_check = 0

# Traverse and decode Braille
for i in range(1, size_rows - 1):
    for j in range(1, size_columns - 1):
        point1 = label_matrix[i][j]
        if point1 != 0 and point1 not in dots_traversed:
            clock = clockwise_traverse(point1)
            a_clock = anticlockwise_traverse(point1)
            c_string = clock[0] if clock[1] >= a_clock[1] else a_clock[0]
            final_string += braille_dict.get(c_string, '')
            space_check = 0
        elif point1 == 0:
            space_check += 1
            if len(final_string) != 0 and space_check > 4 * shortest_x and final_string[-1] != " ":
                final_string += " "

#  Clean and Enhance
final_string = clean_text(final_string)
enhanced_string = enhance_with_llm(f"Fix and format this Braille-decoded text:\n{final_string}")


# Save final enhanced string to output.txt
with open("output.txt", "w") as f:
    f.write(enhanced_string)

print("✅ Enhanced Output:\n", enhanced_string)

# Save as audio using gTTS
from gtts import gTTS
tts = gTTS(text=enhanced_string, lang='en')
tts.save("output.mp3")
print("🔊 Audio saved as output.mp3")


#  Sentiment & NER
sentiment, entities = analyze_sentiment_ner(enhanced_string)

with open("sentiment.txt", "w") as f:
    f.write("Sentiment:\n" + str(sentiment) + "\n\n")
    f.write("Named Entities:\n")
    for ent in entities:
        f.write(str(ent) + "\n")


print("DECODED TEXT:", enhanced_string)
print("Sentiment:", sentiment)
print("Named Entities:", entities)
