from flask import Flask, render_template, request, jsonify
from gensim.models import Word2Vec
import numpy as np

app = Flask(__name__)

# Example corpus of university-related queries
sentences = [
    ["admission", "requirements", "apply", "eligibility"],
    ["deadline", "application", "submit", "date"],
    ["courses", "programs", "degree", "subjects"],
    ["campus", "library", "facilities", "labs"],
    ["scholarship", "fees", "tuition", "financial"],
]

# Train Word2Vec model
model = Word2Vec(sentences, min_count=1, vector_size=50)

# Predefined responses
responses = {
    "admission": "To apply for HN University, submit your transcripts and fill out the online application form.",
    "deadline": "The application deadline for HN University is 31st December each year.",
    "courses": "HN University offers Engineering, Computer Science, Arts, and Business programs.",
    "campus": "Our campus includes a modern library, computer labs, sports facilities, and cafeterias.",
    "scholarship": "HN University offers merit-based and need-based scholarships for eligible students."
}

# Function to find most similar word in vocabulary
def get_response(user_input):
    user_input = user_input.lower().split()
    max_sim = 0
    best_key = None
    for key in responses:
        for word in user_input:
            if word in model.wv.key_to_index:
                sim = model.wv.similarity(word, key)
                if sim > max_sim:
                    max_sim = sim
                    best_key = key
    if best_key:
        return responses[best_key]
    else:
        return "Sorry, I didn't understand that. Please ask about admission, courses, deadlines, or campus facilities."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def chatbot_response():
    user_input = request.form["message"]
    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
