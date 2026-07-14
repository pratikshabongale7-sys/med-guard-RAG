"""Step 3: create chunks of the cleaned text recursively"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# chunk parsed articles with overlapping
def chunk_articles(articles: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""], # split at paras, lines, new lines - broader split first
    )

    chunks = []
    for article in articles:
        text = f"{article['title']}. {article['abstract']}"
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append({
                "id": f"{article['pmid']}-{i}",  # human-readable chunk key
                "text": piece,
                "pmid": article["pmid"],
                "title": article["title"],
                "journal": article["journal"],
                "year": article["year"],
                "url": article["url"],
            })
    return chunks