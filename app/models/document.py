from pydantic import BaseModel


class Document(BaseModel):
    paper_id: str
    chunk_id: str
    title: str
    text: str
    primary_category: str
    published: str