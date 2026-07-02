"""Shared vocabularies and defaults for JD-driven ranking."""

PROFICIENCY_WEIGHT = {
    "beginner": 0.35,
    "intermediate": 0.65,
    "advanced": 0.85,
    "expert": 1.0,
}

KNOWN_SKILL_VOCABULARY = frozenset(
    {
        "python", "java", "javascript", "typescript", "go", "golang", "rust", "c++", "c#",
        "sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
        "nlp", "natural language processing", "pytorch", "tensorflow", "keras",
        "scikit-learn", "machine learning", "deep learning", "data science",
        "fine-tuning llms", "llm", "llms", "rag", "retrieval augmented generation",
        "vector search", "embeddings", "faiss", "milvus", "pinecone", "weaviate",
        "chromadb", "opensearch", "pgvector", "learning to rank", "ltr", "ranking",
        "recommendation systems", "information retrieval", "semantic search",
        "transformers", "hugging face", "bert", "gpt", "openai api", "langchain",
        "llamaindex", "haystack", "mlflow", "mlops", "model deployment",
        "computer vision", "react", "node.js", "angular", "vue", "spring boot",
        "microservices", "api design", "system design", "agile", "scrum",
        "tableau", "power bi", "excel", "salesforce", "sap", "figma",
        "spark", "pyspark", "airflow", "kafka", "etl", "data engineering",
        "devops", "linux", "git", "rest api", "graphql", "prompt engineering",
        "lora", "peft", "xgboost", "pandas", "numpy",
        "ndcg", "mrr", "map", "a/b testing", "ab testing", "offline evaluation",
        "online evaluation", "reciprocal rank", "mean average precision",
        "sentence-transformers", "bge", "e5", "qdrant", "hybrid retrieval",
        "open source", "oss", "github",
        "distributed systems", "model serving", "inference optimization",
        "triton", "vllm", "tensorrt",
    }
)

# Ranking-system evaluation vocabulary (JD must-have section).
EVAL_RANKING_TERMS = frozenset(
    {
        "ndcg", "mrr", "map", "a/b test", "ab test", "a/b testing",
        "offline evaluation", "online evaluation", "offline benchmark",
        "online benchmark", "reciprocal rank", "mean average precision",
        "learning to rank", "ltr", "ranking metric", "evaluation framework",
    }
)

# JD disqualifier: CV/speech/robotics without NLP/IR exposure.
CV_DOMAIN_TERMS = frozenset(
    {
        "computer vision", "speech recognition", "robotics", "autonomous",
        "object detection", "yolo", "slam", "image segmentation", "asr",
    }
)

NLP_IR_DOMAIN_TERMS = frozenset(
    {
        "nlp", "natural language", "retrieval", "ranking", "embedding",
        "llm", "rag", "search", "recommendation", "information retrieval",
        "semantic search", "vector search", "ir ", "text mining",
    }
)

# Pre-LLM-era production ML (JD: understood retrieval/ranking before LangChain era).
PRE_LLM_PRODUCTION_TERMS = frozenset(
    {
        "ranking", "retrieval", "recommendation", "search", "information retrieval",
        "xgboost", "lightgbm", "learning to rank", "ltr", "elasticsearch",
        "collaborative filtering", "matrix factorization", "production deployment",
        "serving", "million users", "shipped", "deployed",
    }
)

# Academic / pure-research environments (JD §5-9y hard disqualifier).
ACADEMIC_RESEARCH_TERMS = frozenset(
    {
        "academic lab", "university lab", "research lab", "research institute",
        "postdoctoral", "postdoc", "phd candidate", "doctoral student",
        "pure research", "published papers", "research scientist",
    }
)

# Senior roles expected to still write code (JD §5-9y disqualifier).
NON_CODING_SENIOR_TITLE_TERMS = frozenset(
    {
        "architect", "tech lead", "technical lead", "engineering manager",
        "principal engineer", "staff engineer", "staff ml", "staff machine learning",
        "senior engineer", "senior ml engineer", "senior ai engineer",
        "senior machine learning", "lead engineer", "lead ml",
    }
)

# Architecture-only work signals (no hands-on code).
ARCHITECTURE_ONLY_TERMS = frozenset(
    {
        "architecture review", "roadmap", "strategy", "stakeholder",
        "org design", "technical direction", "managed a team of",
        "people management", "hiring pipeline",
    }
)

# Recent hands-on code signals (JD: architect who hasn't coded in 18 months).
RECENT_CODE_TERMS = frozenset(
    {
        "implemented", "built", "shipped", "deployed", "production",
        "wrote", "coded", "maintained", "refactored", "python", "pytorch",
        "tensorflow", "serving", "pipeline",
    }
)

EXTERNAL_VALIDATION_TERMS = frozenset(
    {
        "open source", "open-source", "github", "published", "paper",
        "arxiv", "conference", "neurips", "icml", "acl", "emnlp",
        "kaggle", "hugging face hub", "oss contribution",
    }
)

# Backward-compatible alias used by docs/eda
AI_CORE_SKILLS = KNOWN_SKILL_VOCABULARY

DEFAULT_NEGATIVE_ROLES = (
    "hr manager",
    "human resources",
    "accountant",
    "accounting",
    "civil engineer",
    "mechanical engineer",
    "graphic designer",
    "content writer",
    "marketing manager",
    "sales executive",
    "customer support",
    "operations manager",
    "project manager",
    "business analyst",
    "qa engineer",
    "test engineer",
)

# JD corridor emphasis (e.g. Pune/Noida in founding-team JD).
JD_CORRIDOR_CITIES = ("pune", "noida")

# Tier-1 Indian cities JD explicitly welcomes (Hyderabad, Mumbai, Delhi NCR, etc.).
TIER1_WELCOME_CITIES = (
    "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "noida", "pune",
    "bangalore", "bengaluru",
)

# Tier-1 Indian cities for relocation-friendly scoring.
TIER1_RELOC_CITIES = (
    "bangalore", "bengaluru", "mumbai", "hyderabad", "chennai",
    "delhi", "gurgaon", "gurugram", "kolkata", "pune", "noida",
)

KNOWN_LOCATIONS = (
    "pune", "noida", "delhi", "gurgaon", "gurugram", "hyderabad",
    "mumbai", "bangalore", "bengaluru", "chennai", "kolkata", "indore",
    "chandigarh", "kochi", "coimbatore", "vizag", "trivandrum",
    "san francisco", "new york", "london", "singapore", "toronto",
    "sydney", "dubai", "remote",
)

IT_SERVICES_COMPANIES = (
    "tcs", "infosys", "wipro", "cognizant", "accenture",
    "capgemini", "mindtree", "hcl", "tech mahindra", "mphasis",
)

# Product / SaaS employers (boost current-company signal when JD cares about product exp).
KNOWN_PRODUCT_COMPANIES = (
    "swiggy", "zomato", "flipkart", "razorpay", "cred", "meesho", "nykaa",
    "freshworks", "zoho", "inmobi", "ola", "vedantu", "phonepe", "paytm",
    "redrob", "byju", "unacademy", "dream11", "sharechat", "postman",
    "browserstack", "chargebee", "clevertap", "druva", "groww",
)

WORK_MODE_VALUES = ("onsite", "hybrid", "remote", "flexible")

# Career description phrases that contradict common trap titles (honeypot mismatches).
CAREER_MISMATCH_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "operations manager",
        (
            "mechanical engineering",
            "solidworks",
            "creo",
            "ansys",
            "dfm",
            "hardware-product",
            "brand design",
            "packaging design",
            "adobe suite",
            "visual system",
            "rebrand",
            "seo strategy",
            "content writing",
            "editorial calendar",
            "longform articles",
            "freelance writer",
        ),
    ),
    (
        "marketing manager",
        (
            "mechanical engineering",
            "solidworks",
            "creo",
            "ansys",
            "customer support",
            "tier-1",
            "tier-2",
            "support agents",
            "dfm",
            "dfma",
        ),
    ),
    (
        "customer support",
        (
            "mechanical engineering",
            "solidworks",
            "brand design",
            "packaging design",
            "dfm",
        ),
    ),
    (
        "content writer",
        (
            "mechanical engineering",
            "solidworks",
            "customer support",
            "tier-1",
            "tier-2",
        ),
    ),
)

CAREER_AI_KEYWORD_TERMS = frozenset(
    {
        "llm",
        "llms",
        "embedding",
        "embeddings",
        "vector db",
        "vector search",
        "retrieval",
        "rag",
        "fine-tuning",
        "fine tuning",
        "langchain",
        "pinecone",
        "faiss",
        "milvus",
        "ndcg",
        "mrr",
        "sentence-transformers",
        "openai embeddings",
    }
)

JOB_HOP_MIN_JOBS = 4
JOB_HOP_MAX_AVG_MONTHS = 18

# Legacy aliases for EDA script compatibility
ROLE_NEGATIVE_PATTERNS = DEFAULT_NEGATIVE_ROLES
ROLE_POSITIVE_PATTERNS = (
    "engineer", "scientist", "developer", "analyst", "architect",
)
PREFERRED_INDIA_LOCATIONS = KNOWN_LOCATIONS[:11]
CAREER_EVIDENCE_TERMS = KNOWN_SKILL_VOCABULARY
