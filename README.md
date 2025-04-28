# LLM-Chatbot-Domain-HR
HR Assistant - Thammasat University
Welcome to the HR Assistant project, a smart chatbot designed to provide seamless and accurate responses to HR-related queries, specifically tailored for Thammasat University. This application leverages advanced natural language processing and vector database technology to deliver precise answers based on a predefined HR dataset, with fallback to general knowledge when needed.

🌟 Features

Intelligent Query Matching: Uses ChromaDB and SentenceTransformer embeddings to find the most relevant HR questions and answers from the dataset.
Multilingual Support: Powered by the paraphrase-multilingual-MiniLM-L12-v2 model for robust language understanding.
Typhoon API Integration: Generates natural, polite Thai responses using the Typhoon-v2-8b-instruct model.
Interactive UI: Built with Gradio for a user-friendly, web-based chatbot interface.
Contextual Fallback: Provides general answers when no matching data is found, ensuring a smooth user experience.
Persistent Storage: Stores HR data in ChromaDB for fast and efficient retrieval.


📋 Prerequisites
To run this project, ensure you have the following installed:

Python 3.8+
Required Python packages (install via pip):pip install chromadb sentence-transformers requests gradio


A valid API key for the Typhoon API.
A CSV file (hr_data.csv) containing HR data with columns: Question, Answer, Type, Category, Subcategory, Detail, EligibleGroup.


🚀 Getting Started

Clone the Repository:
git clone https://github.com/your-repo/hr-assistant-thammasat.git
cd hr-assistant-thammasat


Install Dependencies:
pip install -r requirements.txt


Prepare the HR Data:

Place your hr_data.csv file in the project root.
Ensure the CSV follows the required format (see Data Format).


Set Up the Typhoon API Key:

Replace the api_key in the script with your Typhoon API key:api_key = "your-typhoon-api-key"




Run the Application:
python hr_assistant.py


This will launch the Gradio interface in your browser.




📊 Data Format
The hr_data.csv file should have the following columns:



Column
Description



Question
The HR-related question


Answer
The corresponding answer


Type
Type of query (e.g., policy, benefit)


Category
Main category (e.g., welfare, leave)


Subcategory
Subcategory (e.g., medical, vacation)


Detail
Additional details (if any)


EligibleGroup
Eligible group (e.g., staff, faculty)


Example:
Question,Answer,Type,Category,Subcategory,Detail,EligibleGroup
"What is the medical benefit?","Coverage up to 50,000 THB annually","Benefit","Welfare","Medical","Outpatient only","All staff"


🛠️ How It Works

Data Loading: The script loads HR data from hr_data.csv and stores it in ChromaDB with embeddings for fast similarity search.
Query Processing:
User inputs a question via the Gradio interface.
The question is embedded and searched against the ChromaDB collection.
If a close match (distance < 0.6) is found, the corresponding answer and metadata are retrieved.


Response Generation:
If a match is found, the Typhoon API generates a polite, natural Thai response using the dataset's answer and metadata.
If no match is found, the API provides a general response based on the query.


User Interaction:
The Gradio UI displays the conversation history.
Users can clear the chat history or submit new questions.




🌐 Deployment
To deploy the application publicly:

Use Gradio's share=True option for temporary hosting:demo.launch(share=True)


For permanent deployment, consider hosting on:
Hugging Face Spaces: Upload the script and dependencies to a Space.
Heroku/Vercel: Deploy as a web app with a Python backend.
Docker: Containerize the app for scalability.




🔧 Troubleshooting

ChromaDB Errors: Ensure the ./hr_db directory is writable and not corrupted.
Typhoon API Issues: Verify your API key and internet connection. Check the API status at Typhoon's website.
Gradio Interface Not Loading: Ensure gradio is installed and try running on a different port:demo.launch(server_port=7861)


CSV Encoding Issues: Use utf-8-sig encoding when saving the CSV file.


🤝 Contributing
We welcome contributions! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.


📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

📬 Contact
For questions or feedback, reach out to the Thammasat University HR team or open an issue on the GitHub repository.

Built with ❤️ for Thammasat University HR Department
