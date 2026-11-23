import re

import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AI Plagiarism Checker MVP", layout="wide")


@st.cache_resource
def load_model():
	# 80 MB lightweight model; downloads once and caches locally
	return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def extract_text(uploaded_file) -> str:
	name = uploaded_file.name.lower()
	if name.endswith(".txt"):
		return uploaded_file.read().decode("utf-8", errors="ignore")
	elif name.endswith(".docx"):
		doc = Document(uploaded_file)
		return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
	elif name.endswith(".pdf"):
		reader = PdfReader(uploaded_file)
		return "\n".join(page.extract_text() or "" for page in reader.pages)
	raise ValueError(
		f"Unsupported file type: {uploaded_file.name}. Upload a .docx, .pdf, or .txt file."
	)


def split_sentences(text: str, min_words: int = 5) -> list[str]:
	# Split on sentence-ending punctuation and drop very short fragments
	raw = re.split(r"(?<=[.!?])\s+", text.strip())
	return [sentence.strip() for sentence in raw if len(sentence.split()) >= min_words]


def check_plagiarism(sub_text: str, ref_docs: dict[str, str], threshold: float):
	sub_sentences = split_sentences(sub_text)

	ref_sentences = []
	ref_sources = []
	for filename, text in ref_docs.items():
		for sentence in split_sentences(text):
			ref_sentences.append(sentence)
			ref_sources.append(filename)

	if not sub_sentences or not ref_sentences:
		return 0.0, []

	model = load_model()
	# Encode both sets into vector embeddings and compute cosine similarity
	sub_embs = model.encode(sub_sentences, convert_to_tensor=True)
	ref_embs = model.encode(ref_sentences, convert_to_tensor=True)
	sim_matrix = util.cos_sim(sub_embs, ref_embs)

	flagged = []
	for i, sub_sentence in enumerate(sub_sentences):
		scores = sim_matrix[i]
		best_idx = int(scores.argmax())
		best_score = float(scores[best_idx])

		if best_score >= threshold:
			flagged.append(
				{
					"submitted": sub_sentence,
					"matched": ref_sentences[best_idx],
					"source": ref_sources[best_idx],
					"similarity": round(best_score * 100, 1),
				}
			)

	overall_pct = round((len(flagged) / len(sub_sentences)) * 100, 1)
	return overall_pct, flagged


# --- Streamlit UI ---
st.title("AI Plagiarism Checker (MVP)")
st.caption("Checks exact copying and AI/manual paraphrasing using semantic embeddings.")

threshold = st.sidebar.slider(
	"Similarity Threshold",
	min_value=0.50,
	max_value=0.95,
	value=0.75,
	step=0.05,
	help="0.75 catches moderate paraphrasing; 0.85+ catches near-exact matches.",
)

col1, col2 = st.columns(2)

with col1:
	st.subheader("1. Document to Check")
	sub_file = st.file_uploader(
		"Upload draft (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"]
	)
	sub_paste = st.text_area("Or paste text directly", height=180)

with col2:
	st.subheader("2. Reference Sources")
	ref_files = st.file_uploader(
		"Upload one or more source files",
		type=["docx", "pdf", "txt"],
		accept_multiple_files=True,
	)

if st.button("Check Originality", type="primary"):
	submitted_text = extract_text(sub_file) if sub_file else sub_paste
	reference_docs = {file.name: extract_text(file) for file in ref_files} if ref_files else {}

	if not submitted_text.strip():
		st.warning("Upload a document or paste text to check.")
	elif not reference_docs:
		st.warning("Upload at least one reference source file.")
	elif not split_sentences(submitted_text):
		st.warning("The submitted text has no sentences with at least five words to analyze.")
	elif not any(split_sentences(text) for text in reference_docs.values()):
		st.warning(
			"None of the reference files contain sentences with at least five words to analyze."
		)
	else:
		with st.spinner("Analyzing semantic similarity..."):
			overall_score, matches = check_plagiarism(
				submitted_text, reference_docs, threshold
			)

		st.divider()
		st.metric("Flagged Similarity", f"{overall_score}%", f"{len(matches)} passages flagged")

		if not matches:
			st.success("No passages exceeded the similarity threshold.")
		else:
			for item in sorted(matches, key=lambda match: match["similarity"], reverse=True):
				with st.expander(f"{item['similarity']}% match — Source: {item['source']}"):
					st.markdown(f"**Your Draft:** {item['submitted']}")
					st.markdown(f"**Matched Source:** {item['matched']}")
