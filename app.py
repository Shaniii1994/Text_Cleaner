import streamlit as st
import wordcloud
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import base64
import time
import spacy
import neattext.functions as nfx
from collections import Counter
from spacy import displacy

matplotlib.use("Agg")

# Function to check and download the model if not available
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        import subprocess
        print(f"Model {model_name} not found. Downloading...")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        nlp = spacy.load(model_name)
        return nlp

# Load NLP model
nlp = load_spacy_model()

timestr = time.strftime("%Y%m%d-%H%M%S")

# Function to get most common tokens
def get_most_common_tokens(docx, num=10):
    word_frequency = Counter(docx.split())
    most_common_tokens = word_frequency.most_common(num)
    return dict(most_common_tokens)

# Function to plot WordCloud
def plot_wordcloud(docx):
    mywordcloud = wordcloud.WordCloud(width=800, height=400, background_color="white").generate(docx)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(mywordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# Text Analyzer
def text_analyzer(my_text):
    docx = nlp(my_text)
    all_data = [
        (token.text, token.shape_, token.pos_, token.tag_, token.lemma_, token.is_alpha, token.is_stop)
        for token in docx
    ]
    df = pd.DataFrame(all_data, columns=["Text", "Shape", "POS", "Tag", "Lemma", "Is Alpha", "Is Stopword"])
    return df

# Named Entity Recognition (NER)
def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(ent.text, ent.label_) for ent in docx.ents]
    return entities

# Render Named Entities
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem;">{}</div>"""

def render_entities(raw_text):
    docx = nlp(raw_text)
    html = displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    return HTML_WRAPPER.format(html)

# Function to download result as CSV
def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = f"nlp_result_{timestr}.csv"
    st.markdown("### **⬇️ Download CSV File**")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click to download!</a>'
    st.markdown(href, unsafe_allow_html=True)

# Function to download cleaned text
def text_downloader(raw_text):
    b64 = base64.b64encode(raw_text.encode()).decode()
    new_filename = f"Clean_text_result_{timestr}.txt"
    st.markdown("### ⬇️ Download File ###")
    href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Click here!</a>'
    st.markdown(href, unsafe_allow_html=True)

# Streamlit App
def main():
    st.title("Welcome to Text Cleaning App 📜")

    menu = ["TextCleaner", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "TextCleaner":
        st.subheader("Text Cleaning")
        text_file = st.file_uploader("Upload Text File", type=["txt"])

        normalize_case = st.sidebar.checkbox("Normalize Case")
        clean_stopwords = st.sidebar.checkbox("Remove Stopwords")
        clean_punctuation = st.sidebar.checkbox("Remove Punctuation")
        clean_emails = st.sidebar.checkbox("Remove Emails")
        clean_special_characters = st.sidebar.checkbox("Remove Special Characters")
        clean_numbers = st.sidebar.checkbox("Remove Numbers")
        clean_urls = st.sidebar.checkbox("Remove URLs")

        if text_file is not None:
            file_details = {"Filename": text_file.name, "Filesize": text_file.size, "Filetype": text_file.type}
            st.write(file_details)

            # Decode file
            raw_text = text_file.read().decode("utf-8")

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("Original Text"):
                    st.write(raw_text)

            with col2:
                with st.expander("Processed Text"):
                    if normalize_case:
                        raw_text = raw_text.lower()
                    if clean_stopwords:
                        raw_text = nfx.remove_stopwords(raw_text)
                    if clean_numbers:
                        raw_text = nfx.remove_numbers(raw_text)
                    if clean_urls:
                        raw_text = nfx.remove_urls(raw_text)
                    if clean_punctuation:
                        raw_text = nfx.remove_punctuation(raw_text)
                    if clean_emails:
                        raw_text = nfx.remove_emails(raw_text)
                    if clean_special_characters:
                        raw_text = nfx.remove_special_characters(raw_text)

                    st.write(raw_text)
                    text_downloader(raw_text)

            # Text Analysis
            with st.expander("Text Analysis"):
                token_results_df = text_analyzer(raw_text)
                st.dataframe(token_results_df)
                make_downloadable(token_results_df)

            # WordCloud Plot
            with st.expander("Plot Wordcloud"):
                plot_wordcloud(raw_text)

            # POS Tags Plot
            with st.expander("Plot POS Tags"):
                fig, ax = plt.subplots()
                sns.countplot(x=token_results_df["POS"], ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

    else:
        st.subheader("About")
        st.write("📜 A simple yet powerful NLP-based tool for cleaning and analyzing text, built with Streamlit. 🚀"
                 "🛠️ Remove stopwords, punctuation, emails, URLs & more—effortlessly! 💡"
                 "📊 Generate word clouds, POS tags, and entity visualizations with ease! 🎨"
                 "💾 Upload your text & let the magic happen! ✨")

if __name__ == "__main__":
    main()
