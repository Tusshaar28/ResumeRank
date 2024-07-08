import os
import streamlit as st
from main import func


st.title("Resume Ranker")
TEMP_DIR = "temp_dir"
os.makedirs(TEMP_DIR, exist_ok=True)

jd = st.text_area("Enter the Job Description")
pdfs = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
top = st.number_input("Number of top resumes to display", min_value=1, step=1, value=1)

if st.button("Rank Resumes"):
    if jd and pdfs:
        pdfiles = []
        for pdf in pdfs:
            data = pdf.read()
            path = os.path.join(TEMP_DIR, pdf.name)
            with open(path, 'wb') as f:
                f.write(data)
            pdfiles.append(path)

        ranks = func(jd, pdfiles)

        st.write(f"Top {top} Resumes:")
        for rank_num, (filename, similarity) in enumerate(ranks[:top], start=1):
            st.write(f"Rank {rank_num}: Filename - {filename}, Similarity: {similarity:.2f}")

        for pdf_path in pdfiles:
            os.remove(pdf_path)
    else:
        st.write("Please enter a job description and upload at least one PDF file.")
