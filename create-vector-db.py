# %%
import os
import duckdb
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dataclasses import dataclass, field
from typing import Optional, List
from uuid import uuid5, NAMESPACE_DNS
from more_itertools import chunked
from tqdm import tqdm

load_dotenv()
NACE_NAMESPACE = uuid5(NAMESPACE_DNS, "nace-rev2")
# %%
client_llmlab = OpenAI(
    base_url=os.environ.get("LLMLAB_URL"),
    api_key=os.environ.get("LLMLAB_API_KEY"),
)
# %% 
models = client_llmlab.models.list()
print(models.data)
# %%
client_qdrant = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
    port=os.environ.get("QDRANT_API_PORT"),
    check_compatibility=False
)
# %%
collections = client_qdrant.get_collections()
print(collections)
# %%
con = duckdb.connect(database=":memory:")

con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

path_nace = 'https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/NACE_Rev2.1_Structure_Explanatory_Notes_EN.tsv'
query_definition = f"SELECT * FROM read_csv('{path_nace}')"
table = con.execute(query_definition).to_arrow_table()
nace = table.to_pylist()

nace[22]
# %%


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    cleaned = " ".join(str(value).replace("\n", " ").split())
    return cleaned or None

@dataclass
class NaceDocument:
    code: str
    title: str
    level: Optional[int] = None
    parent_code: Optional[str] = None
    includes: Optional[str] = None
    includes_also: Optional[str] = None
    excludes: Optional[str] = None
    vector: Optional[List[float]] = None

    text: str = field(init=False)

    @classmethod
    def from_raw(cls, raw: dict, with_includes_also=True, with_excludes=False,) -> "NaceDocument":
        instance = cls(
            code=raw["CODE"].strip(),
            title=_clean(raw["HEADING"]),
            level=raw["LEVEL"],
            parent_code=raw["PARENT_CODE"],
            includes=_clean(raw["Includes"]),
            includes_also=_clean(raw["IncludesAlso"]),
            excludes=_clean(raw["Excludes"]),
            )

        instance.text = instance.to_embedding_text(
                    with_includes_also=with_includes_also,
                    with_excludes=with_excludes,
                )

        return instance

    def to_embedding_text(
        self,
        *,
        with_includes_also: bool = False,
        with_excludes: bool = False
    ) -> str:

        code = f"# Code: {self.code or ''}"
        title = f"# Title: {self.title or ''}"
        includes = f"## Includes: {self.title or ''}"
        parts = [code, title, includes]
        if with_includes_also and self.includes_also:
            includes_also = f"## Includes Also: {self.includes_also}"
            parts.append(includes_also)
        if with_excludes and self.excludes:
            excludes = f"## Excludes: {self.excludes}"
            parts.append(excludes)
        return "\n".join(parts).strip()

    def get_embeddings(self, client_llmlab, model_name) -> List[float]:
        response = client_llmlab.embeddings.create(
            model=model_name,
            input=self.text,
        )
        self.vector = response.data[0].embedding
        return self.vector

    def to_qdrant_point(
        self,
    ) -> PointStruct:

        return PointStruct(
            id=str(uuid5(NACE_NAMESPACE, self.code)),
            vector=self.vector,
            payload={
                "code": self.code,
                "level": self.level,
                "parent_code": self.parent_code,
                "text": self.text,
            }
        )


# %%
# Example to instance a NaceDocument class
animal = NaceDocument.from_raw(raw=nace[22], with_includes_also=True, with_excludes=True)

# %%
instances = []
for code in nace:
    instances.append(NaceDocument.from_raw(code))
# %%
num = 50
exclusion = False
single = NaceDocument.from_raw(raw=nace[num], with_includes_also=True, with_excludes=exclusion)
print(f"Printing index: {num}")
print("=============================================")
print("=============================================")
print(f"Printing text to embed ({'WITH' if exclusion else 'WITHOUT'} exclusions):")
print(single.text)
print("=============================================")
print("=============================================")
# %%
EMB_MODEL_NAME = "qwen3-embedding-8b"
emb_dim = 4096

COLLECTION_NAME = "nace-collection"

# Delete the collection if necessary
if client_qdrant.collection_exists(collection_name=COLLECTION_NAME):
    client_qdrant.delete_collection(collection_name=COLLECTION_NAME)

# Create the collection
client_qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=emb_dim,
        distance=Distance.COSINE
    )
)
# %%

nace_points = []
for nace_code in nace:
    nace_doc = NaceDocument.from_raw(
            raw=nace_code,
            with_includes_also=True,
            with_excludes=True
        )

    nace_doc.get_embeddings(
        client_llmlab,
        EMB_MODEL_NAME,
    )

    nace_points.append(nace_doc.to_qdrant_point())


# %%

BATCH_SIZE = 16
batches = list(chunked(nace_points, BATCH_SIZE))

for batch in tqdm(batches, desc="Uploading to Qdrant", unit="batch"):
    try:
        client_qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )
    except Exception as e:
        tqdm.write(f"✗ Batch failed: {e}")


# %%
