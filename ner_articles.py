import pandas as pd
from camel_tools.ner import NERecognizer
from tqdm import tqdm

# Load the NER model
print("Loading NER model...")
ner = NERecognizer.pretrained()

# Load your articles
print("Loading articles...")
df = pd.read_csv("C:/Users/hp/Documents/DAT505/DAT505-RR2/articles_for_ner.csv")

# Faster function — only processes first 50 words per article
# Celebrity articles mention names early, no need to read entire article
def count_persons(text):
    if not isinstance(text, str):
        return 0
    try:
        # Only take first 50 words — much faster, still captures names
        words = str(text).split()[:100]
        if len(words) == 0:
            return 0
        tags = ner.predict_sentence(words)
        return sum(1 for tag in tags if "PER" in tag)
    except:
        return 0

# Run with progress bar so you can see speed
print("Running NER on articles...")
results = []
for text in tqdm(df["Body"], desc="Processing"):
    results.append(count_persons(text))

df["person_count"] = results
df["person_density"] = df["person_count"] / df["Body"].apply(
    lambda x: len(str(x).split())
)

# Save
df[["doc_id", "person_count", "person_density"]].to_csv(
    "C:/Users/hp/Documents/DAT505/DAT505-RR2/ner_results.csv",
    index=False
)
print("Done! Results saved.")