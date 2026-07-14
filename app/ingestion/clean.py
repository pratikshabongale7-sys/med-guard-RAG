"""Step 2: clean up parsed pubmed articles"""

import re

# collapse whitespace and trim
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()