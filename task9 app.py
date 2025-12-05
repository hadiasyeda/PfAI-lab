from flask import Flask, render_template
import requests

app = Flask(__name__)

JOKE_API_URL = "https://official-joke-api.appspot.com/random_joke"

@app.route("/")
def home():
    # Fetching a joke from the API
    response = requests.get(JOKE_API_URL)

    if response.status_code == 200:
        joke = response.json()
        setup = joke["setup"]
        punchline = joke["punchline"]
    else:
        setup = "Error loading joke."
        punchline = ""

    return render_template("index.html", setup=setup, punchline=punchline)

if __name__ == "__main__":
    app.run(debug=True)
