import numpy as np
import cv2
import os
import re
from gtts import gTTS
from dotenv import load_dotenv
import requests
from transformers import pipeline
import re

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
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    
    # Capitalize the first letter of each sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = ' '.join(s.capitalize() for s in sentences)
    
    return cleaned



def enhance_with_llm(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    try:
        result = response.json()
        if 'choices' in result:
            return result['choices'][0]['message']['content'].strip()
        else:
            error_message = result.get('error', result)
            return f" LLM Error: {error_message}"
    except Exception as e:
        return f" Exception: {str(e)}"

sentiment_pipeline = pipeline("sentiment-analysis")
ner_pipeline = pipeline("ner", grouped_entities=True)

def analyze_sentiment_ner(text):
    sentiment = sentiment_pipeline(text)
    entities = ner_pipeline(text)
    return sentiment, entities

def calculate_distance(pnt1, pnt2):
    return ((pnt1[1] - pnt2[1]) ** 2 + (pnt1[0] - pnt2[0]) ** 2) ** 0.5

def clockwise_traverse(c_label, centers_list, label_matrix, shortest_x, dots_traversed):
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

def anticlockwise_traverse(a_label, centers_list, label_matrix, shortest_x, dots_traversed):
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

def process_braille_image(image_path="Braille.png"):
    im = cv2.imread(image_path, 0)
    if im is None:
        raise FileNotFoundError(f"{image_path} not found")

    im_padded = np.pad(im, 1, "constant", constant_values=255)
    im_padded = (im_padded // 128) * 255
    size_rows, size_columns = im_padded.shape

    black, white = 0, 255
    label_matrix = np.zeros((size_rows, size_columns))
    label = 0
    list_label = {}
    arrange_array = []

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

    centers_list = {}
    for i in range(1, len(list_label) + 1):
        if list_label[i] == i:
            index = np.where(label_matrix == i)
            x_mid = np.ceil(np.mean(index[0]))
            y_mid = np.ceil(np.mean(index[1]))
            centers_list[i] = (x_mid, y_mid)

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

    for i in range(1, size_rows - 1):
        for j in range(1, size_columns - 1):
            point1 = label_matrix[i][j]
            if point1 != 0 and point1 not in dots_traversed:
                clock = clockwise_traverse(point1, centers_list, label_matrix, shortest_x, dots_traversed)
                a_clock = anticlockwise_traverse(point1, centers_list, label_matrix, shortest_x, dots_traversed)
                c_string = clock[0] if clock[1] >= a_clock[1] else a_clock[0]
                final_string += braille_dict.get(c_string, '') or ''
                space_check = 0
            elif point1 == 0:
                space_check += 1
                if len(final_string) != 0 and space_check > 4 * shortest_x and final_string[-1] != " ":
                    final_string += " "

    return clean_text(final_string)


def text_to_speech(text, filename="output.mp3"):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        return filename
    except Exception as e:
        print("TTS generation failed:", str(e))
        return None

if __name__ == "__main__":
    decoded_text = process_braille_image("Braille.png")
    enhanced = enhance_with_llm(f"Fix and format this Braille-decoded text:\n{decoded_text}")
    with open("output.txt", "w") as f:
        f.write(enhanced)

    text_to_speech(enhanced)
    sentiment, entities = analyze_sentiment_ner(enhanced)

    with open("sentiment.txt", "w") as f:
        f.write("Sentiment:\n" + str(sentiment) + "\n\n")
        f.write("Named Entities:\n")
        for ent in entities:
            f.write(str(ent) + "\n")

    print("DECODED TEXT:", enhanced)
    print("Sentiment:", sentiment)
    print("Named Entities:", entities)
