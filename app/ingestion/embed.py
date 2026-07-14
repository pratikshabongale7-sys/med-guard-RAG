"""Step 4: embed each chunk into dense vectors for semantic matching"""

from fastembed import TextEmbedding

_model: TextEmbedding | None = None

# download and load the ONNX embedding model once and reuse it for later runs
def get_model(model_name: str) -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=model_name)

    return _model

# generate 384-dim semantic dense vectors to push to Qdrant
def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    model = get_model(model_name)

    return [vector.tolist() for vector in model.embed(texts)]
