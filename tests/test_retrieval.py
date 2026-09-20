import pytest
from src.utils.config_parser import load_config
from src.retrieval.retriever import ReportRetriever

def test_retriever_search():
    config = load_config("configs/config.yaml")
    retriever = ReportRetriever(config=config)
    results = retriever.search_similar_reports("cardiomegaly pleural effusion", top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2
    if len(results) > 0:
        assert "findings" in results[0] or "full_report" in results[0]
