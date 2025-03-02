# Customer Data Chatbot

A powerful **AI-powered Chatbot** designed to assist users with **Customer Data Platforms (CDPs)** such as **Segment, mParticle, Lytics, and Zeotap** by extracting, processing, and retrieving relevant documentation-based answers in real-time. 🚀

## 🌟 Features

✅ **Supports Four Major CDPs** – Answers questions related to Segment, mParticle, Lytics, and Zeotap.
✅ **Smart Documentation Retrieval** – Extracts and preprocesses relevant information from official docs.
✅ **NLP-Powered Query Handling** – Understands and responds to varied question formats.
✅ **Semantic Search with BERT** – Uses embeddings to fetch the most relevant documentation snippets.
✅ **Real-Time Interactions** – Supports WebSockets for instant query responses.
✅ **User Authentication** – Secure access to chatbot features.
✅ **Bulma CSS for UI** – Clean and responsive frontend design.
✅ **FastAPI Backend** – High-performance API for fast responses.

## 🛠️ Tech Stack

### **Frontend**
- **React.js** – Dynamic and interactive UI.
- **Bulma CSS** – Elegant and responsive styling.
- **WebSockets** – Real-time chatbot interactions.

### **Backend**
- **FastAPI** – Lightweight and high-performance API.
- **BERT-based Embeddings** – For intelligent query understanding.
- **BAAI/bge-large-en** – For embedding.
- **FAISS / Cosine Similarity** – Efficient semantic search.
- **RESTful API** – Smooth frontend-backend communication.

## 🚀 Setup & Installation

### **1. Clone the Repository**
```sh
 git clone https://github.com/chahat-srivastava/chatbot-assignment.git
 cd customer-data-chatbot
```

### **2. Install Dependencies**
```sh
# Install frontend dependencies
cd chatbot-cdp
npm install

# Install backend dependencies
cd ../backend
pip install -r requirements.txt
```

### **3. Configure Environment Variables**
Create a `.env` file in the backend and frontend directories with the required credentials.

### **4. Start the Application**
```sh
# Start Backend
cd backend
uvicorn main:app --reload

# Start Frontend
cd chatbot-cdp
npm start
```

### **5. Open in Browser**
Visit `http://localhost:3000` to interact with your chatbot.

## 📌 Future Enhancements
- 🔹 Multi-language support.
- 🔹 Integration with additional CDPs.
- 🔹 Voice-based query handling.
- 🔹 Enhanced answer summarization.

---
_Developed with ❤️ by Chahat Srivastava._

