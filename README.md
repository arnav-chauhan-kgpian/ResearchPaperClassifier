Real-Time Autonomous Research Paper Categorization Using Retrieval-Augmented Generation (RAG) and Pathway Services
Overview

This project is a comprehensive research paper classification system which incorporates Machine Learning, Retrieval-Augmented Generation (RAG), and an interactive Streamlit-based UI are combined to classify research papers into their target conferences while providing detailed reasoning for the classification.

Key Features

    -->Streamlit-Based User Interface
        Upload research paper PDFs via an intuitive web application.
        Receive real-time conference classification and detailed rationale.

    -->Machine Learning Classification
        Trained classifiers like Logistic Regression, SVM, and Naive Bayes for initial publishability checks.
        Used a voting classifier for robust predictions.

    -->Retrieval-Augmented Generation (RAG)
        Integrated Pathway’s Vector Store for semantic similarity search, dynamically retrieving relevant reference papers.
        Utilized GeminiEmbedder for creating high-quality embeddings and TokenCountSplitter for efficient chunking.
        Enabled real-time updates of reference papers via the Google Drive Connector.

    -->AI-Powered Conference Classification
        Integrated Gemini AI to classify papers and generate explanations (rationale) for transparency and accuracy.

    -->Multithreaded Architecture
        Developed a multithreaded Vector Store Server to ensure seamless querying and retrieval operations without interruptions.

Technologies and Skills
Key Skills:

    Python: Backend programming for machine learning models and preprocessing.
    Streamlit: User-friendly web-based application development.
    Machine Learning: Ensemble voting classifiers, Logistic Regression, Naive Bayes, and SVM.
    Retrieval-Augmented Generation (RAG): Dynamic retrieval pipelines with reasoning generation.
    Pathway Framework: Advanced semantic search using Vector Store.
    Google Drive API: Integration for live document updates.
    Multithreading: Ensured efficient operations without downtime.

How It Works

    1. Upload a Research Paper:
    Use the Streamlit-based interface to upload your research paper in PDF format.

    2. Preprocessing and Publishability Check:
    The system preprocesses the uploaded paper and predicts publishability using trained machine learning classifiers.

    3. Semantic Retrieval:
    Relevant reference papers are retrieved using Pathway’s Vector Store based on semantic similarity.

    4. Classification and Rationale:
    The paper is classified into one of the target conferences (e.g., CVPR, EMNLP, NeurIPS, etc.) using RAG. The rationale is generated using Gemini AI for better transparency.

    5. Results Display:
    The Streamlit UI displays the classification and reasoning in real-time for user review.

Installation and Setup

Clone the Repository:

git clone https://github.com/arnav-chauhan-kgpian/ResearchPaperClassifier.git cd WebApp

Install Dependencies:

pip install -r requirements.txt

Set Up Pathway Vector Store:

    Ensure Pathway is installed and configured.
    Update the Google Drive credentials file (credentials.json).

Run the Application:

streamlit run app.py

    Access the UI: Open your browser and go to http://localhost:8501 to use the application.

Folder Structure

The repository is organized into two main directories for clarity and separation of concerns:
1. Task

    Contains the Jupyter Notebook (project_final.ipynb) outlining the steps, experiments, and analysis for the competition tasks.

Files and Folders:

    project_final.ipynb: Notebook detailing the problem-solving approaches and methodologies for Task 1 and Task 2.
    Reference/: Folder containing reference research papers used for initial testing and development.
    Papers/: Folder containing different type of reasearch papers for testing the model
    Samples/: Folder containing sample reasearch papers for testing the model

2.  WebApp

    Houses the Streamlit-based application and core logic for research paper classification and RAG-based conference prediction.

Files and Folders:

    app.py: Main Streamlit application file to run the UI.
    model.py: Code for machine learning models and preprocessing logic.
    Reference/: Folder containing dynamically updated reference research papers for Pathway’s Vector Store.
    requirements.txt: Python dependencies for the project.
    uploaded_files/: Stored the uploaded papers in the webapp locally.

Other Files

    README.md: Comprehensive documentation of the project, including setup instructions and workflow.

Creator

This project was developed by:

    Arnav Chauhan
    Undergraduate Student, Department of Mechanical Engineering, IIT Kharagpur
