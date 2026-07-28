"""Step 1: fetch pubmed articles by their IDs and parse them"""

import xml.etree.ElementTree as ET
import httpx2

from app.config import settings

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def append_params(params: dict, api_key: str, email: str) -> dict:
    params["api_key"] = api_key
    params["email"] = email

    return params

# returns a list of PubMed IDs matching the query
def search_pubmed(query: str, max_results: int, api_key: str, email: str) -> list[str]:
    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    if api_key and email:
        params = append_params(params, api_key, email)

    response = httpx2.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    if response.status_code >= 400:
        print("NCBI error body:", response.text[:500])
    response.raise_for_status() # raise any errors with the search
    return response.json()["esearchresult"]["idlist"]


# fetch full articles for the PMIDs - abstracts and metadata
def fetch_abstracts(pmids: list[str], api_key=settings.ncbi_api_key, email=settings.ncbi_email, batch_size=100):
    if not pmids:
        return []
    pmids = [p for p in pmids if p and str(p).isdigit()]   # drop empty/non-numeric ids
    articles = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        params = {"db": "pubmed", "id": ",".join(batch),
                  "rettype": "abstract", "retmode": "xml"}
        if api_key:
            params["api_key"] = api_key
        if email:
            params["email"] = email
        resp = httpx2.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=60)
        resp.raise_for_status()
        articles.extend(_parse_articles(resp.text))
    return articles

#
def _parse_articles(xml_response: str) -> list[dict]:
    root = ET.fromstring(xml_response)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""
        title = article.findtext(".//ArticleTitle") or ""
        parts = [t.text or "" for t in article.findall(".//AbstractText")]
        abstract = " ".join(p for p in parts if p).strip()

        if not abstract:
            continue
        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": article.findtext(".//Journal/Title") or "",
            "year": article.findtext(".//PubDate/Year") or "",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return articles

