
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from chatbot import Chatbot  # Import your chatbot class

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load chatbot with data
data_files = [r"C:\Users\Varun\Downloads\greetings.csv",r"C:\Users\Varun\Downloads\pythonfinal.csv",r"C:\Users\Varun\Downloads\ai.csv",r"C:\Users\Varun\Downloads\os.csv",r"C:\Users\Varun\Downloads\json.json"]  # Ensure the correct filename
bot = Chatbot(data_files)

@app.route("/home", methods=["POST"])
def home():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Invalid request"}), 400

    user_input = data["message"]
    response = bot.get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run (host="0.0.0.0",port=5000,debug=True)  # Open to all devices