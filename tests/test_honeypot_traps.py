"""Honeypot trap tests."""

from src.honeypot import (
    is_architect_without_recent_code,
    is_consulting_only_career,
    is_cv_without_nlp_ir,
    is_langchain_only_profile,
    is_research_only_profile,
    lacks_external_validation,
)
from src.jd_parser import parse_job_description


def test_langchain_only_trap():
    skills = [{"name": "LangChain", "proficiency": "expert", "duration_months": 6}]
    career = [{"title": "AI Tinkerer", "description": "Built demos with LangChain and OpenAI API."}]
    assert is_langchain_only_profile(skills, career, "")


def test_langchain_with_production_not_trap():
    skills = [{"name": "LangChain", "proficiency": "advanced", "duration_months": 24}]
    career = [
        {
            "title": "ML Engineer",
            "description": "Shipped production ranking pipeline serving millions of users with PyTorch.",
        }
    ]
    assert not is_langchain_only_profile(skills, career, "")


def test_research_only_trap():
    career = [{"title": "Research Scientist", "description": "Published papers on novel architectures."}]
    assert is_research_only_profile("Research Scientist", career, "Academic lab work.")


def test_research_with_production_not_trap():
    career = [
        {
            "title": "Research Scientist",
            "description": "Deployed production retrieval system serving 10M queries.",
        }
    ]
    assert not is_research_only_profile("Applied Scientist", career, "")


def test_consulting_only_trap():
    career = [
        {"company": "TCS", "description": "Client delivery project.", "duration_months": 36},
        {"company": "Infosys", "description": "Maintenance project.", "duration_months": 40},
    ]
    assert is_consulting_only_career(career)


def test_consulting_with_product_history_not_trap():
    career = [
        {"company": "TCS", "description": "Internal tools.", "duration_months": 24},
        {
            "company": "Razorpay",
            "description": "Shipped recommendation system at marketplace product.",
            "duration_months": 30,
        },
    ]
    assert not is_consulting_only_career(career)


def test_cv_without_nlp_trap():
    skills = [{"name": "YOLO", "proficiency": "expert"}]
    career = [{"title": "CV Engineer", "description": "Object detection models."}]
    assert is_cv_without_nlp_ir("Computer Vision Engineer", career, skills, "")


def test_architect_no_recent_code_trap():
    career = [
        {
            "title": "Principal Architect",
            "description": "Led architecture reviews and roadmaps.",
            "duration_months": 24,
            "is_current": True,
        }
    ]
    assert is_architect_without_recent_code("Principal Architect", career, "Strategy and planning.")


def test_senior_engineer_no_recent_code_trap():
    career = [
        {
            "title": "Senior Engineer",
            "description": "Technical direction, roadmap, and stakeholder alignment.",
            "duration_months": 24,
            "is_current": True,
        }
    ]
    assert is_architect_without_recent_code("Senior Engineer", career, "Org design and hiring.")


def test_senior_engineer_with_recent_code_not_trap():
    career = [
        {
            "title": "Senior ML Engineer",
            "description": "Implemented production ranking pipeline in Python and PyTorch.",
            "duration_months": 18,
            "is_current": True,
        }
    ]
    assert not is_architect_without_recent_code("Senior ML Engineer", career, "")


def test_langchain_pre_llm_salvage():
    skills = [{"name": "LangChain", "proficiency": "advanced", "duration_months": 8}]
    career = [
        {
            "title": "ML Engineer",
            "description": (
                "Built learning-to-rank models and Elasticsearch retrieval before LLM work. "
                "Recent LangChain demos."
            ),
        }
    ]
    assert not is_langchain_only_profile(skills, career, "")


def test_lacks_external_validation_trap():
    sig = {"github_activity_score": 5, "saved_by_recruiters_30d": 0, "endorsements_received": 2}
    career = [{"description": "Proprietary internal systems only."}]
    assert lacks_external_validation(sig, career, 6.0)


def test_framework_enthusiast_trap():
    from src.honeypot import is_framework_enthusiast_profile

    skills = [{"name": "LangChain", "proficiency": "expert", "duration_months": 8}]
    career = [{"title": "AI Hobbyist", "description": "Blog post: how I used LangChain to build a demo."}]
    assert is_framework_enthusiast_profile(skills, career, "Tutorial writer for LangChain projects.")


def test_framework_enthusiast_with_ranking_not_trap():
    from src.honeypot import is_framework_enthusiast_profile

    skills = [{"name": "LangChain", "proficiency": "advanced", "duration_months": 30}]
    career = [
        {
            "title": "ML Engineer",
            "description": "Owned production ranking and retrieval at scale with hybrid search.",
        }
    ]
    assert not is_framework_enthusiast_profile(skills, career, "Shipped recommendation system.")
