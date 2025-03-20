# Research Paper Classifier with Retrieval-Augmented Generation (RAG)

## Overview
This project is a comprehensive research paper classification system developed during the **Kharagpur Data Science Hackathon 2025**, where our team **emerged as one of the top 10 teams** out of 10,000 participants after three competitive rounds. The system combines Machine Learning, Retrieval-Augmented Generation (RAG), and an interactive Streamlit-based UI to classify research papers into their target conferences and provide reasoning for the classification.

---

## Key Features

1. **Streamlit-Based User Interface**  
   - Upload research paper PDFs via an intuitive web application.
   - Receive conference classification and rationale in real-time.

2. **Machine Learning Classification**  
   - Implemented traditional supervised learning models like Logistic Regression, SVM, and Naive Bayes for initial publishability checks.
   - Leveraged voting classifiers for robust predictions.

3. **Retrieval-Augmented Generation (RAG)**  
   - Integrated **Pathway's Vector Store** for semantic similarity search to retrieve relevant reference papers dynamically.
   - Utilized **GeminiEmbedder** for document embedding and **TokenCountSplitter** for efficient chunking.
   - Incorporated **Google Drive Connector** to stream and manage reference research papers, ensuring an updated knowledge base for classification.

4. **AI-Powered Conference Classification**  
   - Embedded **Gemini AI** for providing conference classifications along with explanations (rationale) to ensure transparency and reliability.

5. **Multithreaded Architecture**  
   - Designed a multithreaded Pathway Vector Store Server to handle seamless retrieval and querying operations without interruptions.

---

## Technologies and Skills
### Key Skills:
- **Python**: Core programming for machine learning models, data preprocessing, and backend functionality.
- **Streamlit**: Development of a web-based user interface.
- **Machine Learning**: Logistic Regression, Naive Bayes, SVM, and ensemble voting classifiers.
- **Retrieval-Augmented Generation (RAG)**: Dynamic retrieval and generation pipeline for classification and reasoning.
- **Pathway Framework**: Semantic similarity search with Pathway’s Vector Store.
- **Google Drive API**: Integration to manage reference documents dynamically.
- **Multithreading**: To ensure efficient and continuous operations of the vector store server.

---

## Hackathon Details
### **Kharagpur Data Science Hackathon 2025**
- **Organized by:** IIT Kharagpur
- **Participants:** 10,000+ teams across multiple rounds.
- **Achievement:** Emerged as one of the **top 10 teams** in the final round after a rigorous evaluation of our innovative and technical solution.

---

## How It Works
1. **Upload a Research Paper**: Users upload their research paper in PDF format using the Streamlit-based interface.
2. **Preprocessing and Publishability Check**: The system preprocesses the uploaded paper and predicts whether it is publishable using trained machine learning classifiers.
3. **Semantic Retrieval**: Relevant reference papers are retrieved using Pathway’s Vector Store based on semantic similarity.
4. **Classification and Rationale**: The paper is classified into one of the target conferences (e.g., CVPR, EMNLP, NeurIPS, etc.) using RAG. The classification rationale is generated using Gemini AI.
5. **Results Display**: The results, including classification and reasoning, are displayed in the Streamlit UI.

---

## Installation and Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/research-paper-classifier-rag.git
   cd research-paper-classifier-rag
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Pathway Vector Store:**
   - Ensure Pathway is installed and configured.
   - Update the Google Drive credentials file (`credentials.json`).

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## Folder Structure
- **Reference/**: Contains reference research papers for retrieval.
- **app.py**: Main Streamlit application.
- **pipeline.py**: Core logic for RAG pipeline and machine learning classification.
- **requirements.txt**: Python dependencies.

---

## Future Improvements
- Integration of Hybrid Index (BM25 + Vector Search) for enhanced retrieval performance.
- Improved scalability for larger datasets.
- Enhanced reasoning capabilities with more advanced language models.

---

## Acknowledgments
We extend our gratitude to IIT Kharagpur and the organizers of the **Kharagpur Data Science Hackathon 2025** for providing an excellent platform to showcase and enhance our technical skills.
