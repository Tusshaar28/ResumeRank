# ResumeRank

A simple Resume Ranked App based on Natural Language Processing.

## Features

- Extracts text from PDF resumes
- Preprocesses job descriptions and resumes by tokenizing, removing stopwords, and lemmatizing
- Custom tokenizer to handle special characters in technical skills (e.g., C++, Python)
- Uses CountVectorizer for text vectorization
- Computes cosine similarity to rank resumes based on relevance to the job description

## Project Structure

- `main.py`: Contains the core logic for text extraction, preprocessing, vectorization, and similarity computation.
- `App.py` : Streamlit App
- `requirements.txt`: Lists all the dependencies required for the project.
- `README.md`: Project documentation.



Deployed App : https://resumerankfree.streamlit.app/
