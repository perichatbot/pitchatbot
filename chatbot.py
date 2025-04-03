import pandas as pd
import os
import re
import json
import numpy as np
import faiss
import operator
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

# Load an optimized sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

class Chatbot:
    def __init__(self, files):
        self.data = self.load_data(files)
        if not self.data.empty:
            self.index, self.embeddings = self.build_faiss_index(self.data['Question'])
        else:
            self.index, self.embeddings = None, None

    def load_data(self, files):
        all_data = []
        for file in files:
            if not os.path.exists(file):
                print(f"⚠ Warning: File '{file}' not found.")
                continue
            ext = os.path.splitext(file)[-1].lower()
            try:
                if ext == '.csv':
                    df = pd.read_csv(file)
                elif ext == '.json':
                    df = pd.read_json(file)
                elif ext == '.txt':
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = [line.strip().split('|') for line in f.readlines()]
                        df = pd.DataFrame(lines, columns=['Question', 'Response'])
                else:
                    print(f"⚠ Unsupported file format: {file}")
                    continue

                if {'Question', 'Response'}.issubset(df.columns):
                    all_data.append(df)
                else:
                    print(f"⚠ Warning: '{file}' is missing required columns.")
            except Exception as e:
                print(f"❌ Error loading '{file}': {e}")

        if all_data:
            final_data = pd.concat(all_data, ignore_index=True)
            print(f"✅ Successfully loaded {len(final_data)} questions from all sources.")
            return final_data
        else:
            return pd.DataFrame(columns=['Question', 'Response'])

    def clean_text(self, text):
        return re.sub(r'[^a-zA-Z0-9\s+\-*/().]', '', str(text)).lower().strip()

    def build_faiss_index(self, questions):
        if questions.empty:
            return None, None
        clean_questions = questions.astype(str).apply(self.clean_text).tolist()
        embeddings = model.encode(clean_questions, convert_to_numpy=True)
        if embeddings.shape[0] == 0:
            return None, None
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, embeddings

    def is_math_expression(self, text):
        return bool(re.match(r'^[0-9+\-*/(). ]+$', text))

    def solve_math_expression(self, expression):
        try:
            return eval(expression, {"__builtins__": None}, {})
        except Exception:
            return "❌ Invalid mathematical expression."

    def get_response(self, user_input):
        user_input = self.clean_text(user_input)
        
        if self.is_math_expression(user_input):
            return f"🧮 Result: {self.solve_math_expression(user_input)}"
        
        if self.index is None or self.data.empty:
            return "⚠ No data available to answer your question."
        
        input_embedding = model.encode([user_input], convert_to_numpy=True)
        _, best_match_idx = self.index.search(input_embedding, 3)
        best_match_idx = best_match_idx[0]
        
        responses = []
        for idx in best_match_idx:
            if idx < len(self.data):
                question = self.data.iloc[idx]['Question']
                response = self.data.iloc[idx]['Response']
                similarity_score = fuzz.ratio(user_input, self.clean_text(question))
                responses.append((response, similarity_score))
        
        responses = sorted(responses, key=lambda x: x[1], reverse=True)
        return responses[0][0] if responses else "❌ Sorry, I don't understand that question."

    def chat(self):
        print("🤖 Hello! I am your chatbot. Ask me anything! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye! 👋")
                break
            response = self.get_response(user_input)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    data_files = [r"C:\Users\Varun\Downloads\greetings.csv",r"C:\Users\Varun\Downloads\pythonfinal.csv",r"C:\Users\Varun\Downloads\ai.csv",r"C:\Users\Varun\Downloads\os.csv",r"C:\Users\Varun\Downloads\json.json"]
    bot = Chatbot(data_files)
    bot.chat()
