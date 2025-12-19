"""
Comprehensive Domain Taxonomy for Career Recommendation System
Defines 35+ career domains with detailed metadata
"""

DOMAIN_TAXONOMY = {
    # ==================== TECHNOLOGY (7 domains) ====================
    "data_science_ai": {
        "name": "Data Science & AI",
        "description": "Data analysis, machine learning, artificial intelligence, and analytics",
        "keywords": ["data", "ai", "ml", "machine learning", "analytics", "statistics", "deep learning", 
                    "neural networks", "algorithms", "python", "modeling", "predictions", "insights"],
        "target_careers": 20
    },
    
    "software_development": {
        "name": "Software Development",
        "description": "Software engineering, web/mobile development, and programming",
        "keywords": ["software", "programming", "coding", "development", "web", "mobile", "app", 
                    "javascript", "java", "python", "frontend", "backend", "fullstack", "api"],
        "target_careers": 20
    },
    
    "cybersecurity_it": {
        "name": "Cybersecurity & IT",
        "description": "Information security, network administration, IT support, and cyber defense",
        "keywords": ["security", "cybersecurity", "network", "it support", "firewall", "penetration testing",
                    "vulnerability", "encryption", "infrastructure", "systems", "helpdesk", "cisco"],
        "target_careers": 18
    },
    
    "cloud_devops": {
        "name": "Cloud & DevOps",
        "description": "Cloud computing, DevOps, infrastructure automation, and site reliability",
        "keywords": ["cloud", "aws", "azure", "devops", "kubernetes", "docker", "automation", "ci/cd",
                    "infrastructure", "terraform", "jenkins", "monitoring", "sre", "deployment"],
        "target_careers": 15
    },
    
    "hardware_electronics": {
        "name": "Hardware & Electronics",
        "description": "Hardware engineering, embedded systems, IoT, and electronics design",
        "keywords": ["hardware", "electronics", "embedded", "iot", "microcontroller", "circuit", "pcb",
                    "firmware", "fpga", "arduino", "raspberry pi", "robotics", "sensors"],
        "target_careers": 15
    },
    
    "gaming_xr": {
        "name": "Gaming & Extended Reality",
        "description": "Game development, VR/AR, metaverse, and interactive entertainment",
        "keywords": ["game", "gaming", "unity", "unreal", "vr", "ar", "metaverse", "3d", "graphics",
                    "animation", "gameplay", "virtual reality", "augmented reality", "game design"],
        "target_careers": 15
    },
    
    "telecommunications": {
        "name": "Telecommunications",
        "description": "Telecom networks, wireless systems, 5G, and communication infrastructure",
        "keywords": ["telecom", "telecommunications", "5g", "wireless", "network", "radio", "satellite",
                    "fiber optic", "mobile networks", "communication", "signal processing", "rf"],
        "target_careers": 12
    },
    
    # ==================== HEALTHCARE & WELLNESS (6 domains) ====================
    "medical_clinical": {
        "name": "Medical & Clinical",
        "description": "Physicians, surgeons, nurses, and direct patient care professionals",
        "keywords": ["medical", "doctor", "physician", "surgeon", "nurse", "clinical", "patient care",
                    "hospital", "medicine", "diagnosis", "treatment", "healthcare", "emergency"],
        "target_careers": 25
    },
    
    "mental_health_therapy": {
        "name": "Mental Health & Therapy",
        "description": "Psychology, counseling, psychiatry, and mental wellness services",
        "keywords": ["psychology", "therapy", "counseling", "mental health", "psychiatry", "psychotherapy",
                    "behavioral health", "emotional support", "counselor", "psychologist", "wellness"],
        "target_careers": 15
    },
    
    "allied_health": {
        "name": "Allied Health Services",
        "description": "Physical therapy, occupational therapy, medical technology, and support services",
        "keywords": ["physical therapy", "occupational therapy", "radiology", "medical lab", "respiratory",
                    "speech therapy", "rehabilitation", "diagnostic", "imaging", "ultrasound", "therapy"],
        "target_careers": 20
    },
    
    "public_health": {
        "name": "Public Health & Epidemiology",
        "description": "Population health, disease prevention, epidemiology, and health policy",
        "keywords": ["public health", "epidemiology", "disease prevention", "health policy", "population health",
                    "infectious disease", "vaccination", "health education", "community health", "outbreak"],
        "target_careers": 12
    },
    
    "pharmaceutical_biotech": {
        "name": "Pharmaceutical & Biotech",
        "description": "Drug development, pharmaceutical research, biotechnology, and medical devices",
        "keywords": ["pharmaceutical", "biotech", "drug development", "clinical trials", "medication",
                    "pharmacology", "medical devices", "biopharmaceutical", "genetics", "molecular biology"],
        "target_careers": 15
    },
    
    "alternative_wellness": {
        "name": "Alternative Medicine & Wellness",
        "description": "Holistic health, nutrition, fitness, wellness coaching, and complementary medicine",
        "keywords": ["wellness", "nutrition", "fitness", "holistic", "naturopathy", "acupuncture",
                    "yoga", "massage", "wellness coaching", "alternative medicine", "health coach"],
        "target_careers": 15
    },
    
    # ==================== BUSINESS & FINANCE (6 domains) ====================
    "corporate_management": {
        "name": "Corporate Management",
        "description": "Business management, operations, project management, and executive leadership",
        "keywords": ["management", "operations", "project management", "business", "leadership", "executive",
                    "strategy", "planning", "coordination", "agile", "scrum", "organizational"],
        "target_careers": 20
    },
    
    "entrepreneurship": {
        "name": "Entrepreneurship & Startups",
        "description": "Starting and growing businesses, venture capital, and innovation",
        "keywords": ["entrepreneur", "startup", "founder", "venture capital", "innovation", "business owner",
                    "small business", "angel investor", "pitch", "bootstrapping", "scaling"],
        "target_careers": 12
    },
    
    "finance_investment": {
        "name": "Finance & Investment",
        "description": "Investment banking, wealth management, trading, and financial markets",
        "keywords": ["finance", "investment", "banking", "trading", "wealth management", "portfolio",
                    "stocks", "bonds", "hedge fund", "private equity", "financial markets", "investor"],
        "target_careers": 18
    },
    
    "accounting_audit": {
        "name": "Accounting & Audit",
        "description": "Accounting, auditing, tax, bookkeeping, and financial compliance",
        "keywords": ["accounting", "audit", "tax", "bookkeeping", "cpa", "financial reporting", "gaap",
                    "compliance", "forensic accounting", "payroll", "accounts payable", "ledger"],
        "target_careers": 15
    },
    
    "sales_business_dev": {
        "name": "Sales & Business Development",
        "description": "Sales, business development, account management, and revenue generation",
        "keywords": ["sales", "business development", "account management", "revenue", "deals", "clients",
                    "prospecting", "negotiation", "crm", "lead generation", "closing", "customer"],
        "target_careers": 18
    },
    
    "consulting_strategy": {
        "name": "Consulting & Strategy",
        "description": "Management consulting, strategy, advisory, and organizational transformation",
        "keywords": ["consulting", "strategy", "advisory", "management consulting", "transformation",
                    "business strategy", "change management", "mckinsey", "bain", "bcg", "analysis"],
        "target_careers": 15
    },
    
    # ==================== CREATIVE & MEDIA (5 domains) ====================
    "design_visual_arts": {
        "name": "Design & Visual Arts",
        "description": "Graphic design, UI/UX, product design, and visual creativity",
        "keywords": ["design", "graphic design", "ui", "ux", "visual", "creative", "branding", "illustration",
                    "typography", "layout", "adobe", "figma", "photoshop", "art direction"],
        "target_careers": 20
    },
    
    "writing_publishing": {
        "name": "Writing & Publishing",
        "description": "Writing, editing, journalism, content creation, and publishing",
        "keywords": ["writing", "content", "journalism", "editor", "author", "copywriting", "blogging",
                    "publishing", "technical writing", "creative writing", "editing", "articles"],
        "target_careers": 18
    },
    
    "film_video_production": {
        "name": "Film & Video Production",
        "description": "Filmmaking, video production, cinematography, and post-production",
        "keywords": ["film", "video", "production", "cinematography", "director", "editing", "camera",
                    "filming", "post-production", "premiere", "videography", "documentary", "movie"],
        "target_careers": 18
    },
    
    "music_audio": {
        "name": "Music & Audio",
        "description": "Music production, audio engineering, sound design, and music performance",
        "keywords": ["music", "audio", "sound", "production", "recording", "mixing", "mastering",
                    "composer", "musician", "sound design", "audio engineering", "studio", "dj"],
        "target_careers": 15
    },
    
    "performing_arts": {
        "name": "Performing Arts",
        "description": "Theater, dance, performance, entertainment, and live arts",
        "keywords": ["theater", "acting", "dance", "performance", "stage", "choreography", "broadway",
                    "performer", "entertainment", "drama", "ballet", "contemporary dance", "opera"],
        "target_careers": 15
    },
    
    # ==================== EDUCATION & TRAINING (3 domains) ====================
    "academic_education": {
        "name": "Academic Education",
        "description": "K-12 teaching, higher education, curriculum development, and academic administration",
        "keywords": ["teaching", "education", "teacher", "professor", "curriculum", "classroom", "academic",
                    "instruction", "pedagogy", "learning", "school", "university", "students"],
        "target_careers": 18
    },
    
    "corporate_training": {
        "name": "Corporate Training & Development",
        "description": "Employee training, professional development, and organizational learning",
        "keywords": ["training", "professional development", "corporate training", "learning", "workshop",
                    "employee development", "skills training", "coaching", "facilitation", "leadership development"],
        "target_careers": 12
    },
    
    "educational_technology": {
        "name": "Educational Technology",
        "description": "E-learning, instructional design, educational software, and learning platforms",
        "keywords": ["edtech", "elearning", "instructional design", "online learning", "lms", "educational technology",
                    "digital learning", "course development", "learning platforms", "mooc", "educational software"],
        "target_careers": 12
    },
    
    # ==================== ENGINEERING & MANUFACTURING (5 domains) ====================
    "mechanical_industrial": {
        "name": "Mechanical & Industrial Engineering",
        "description": "Mechanical design, manufacturing, automation, and industrial systems",
        "keywords": ["mechanical", "industrial", "manufacturing", "automation", "cad", "design", "machinery",
                    "production", "assembly", "robotics", "process engineering", "lean manufacturing"],
        "target_careers": 18
    },
    
    "civil_construction": {
        "name": "Civil & Construction Engineering",
        "description": "Civil engineering, construction, infrastructure, and structural design",
        "keywords": ["civil engineering", "construction", "infrastructure", "structural", "building", "roads",
                    "bridges", "architecture", "surveying", "urban planning", "concrete", "project management"],
        "target_careers": 18
    },
    
    "electrical_electronics_eng": {
        "name": "Electrical & Electronics Engineering",
        "description": "Electrical systems, power engineering, electronics, and control systems",
        "keywords": ["electrical", "electronics", "power", "circuits", "control systems", "automation",
                    "electrical engineering", "power systems", "generators", "renewable energy", "grid"],
        "target_careers": 15
    },
    
    "chemical_materials": {
        "name": "Chemical & Materials Engineering",
        "description": "Chemical engineering, materials science, process engineering, and polymer science",
        "keywords": ["chemical", "materials", "polymer", "process engineering", "chemistry", "petrochemical",
                    "materials science", "nanotechnology", "composites", "chemical processes", "lab"],
        "target_careers": 15
    },
    
    "aerospace_automotive": {
        "name": "Aerospace & Automotive Engineering",
        "description": "Aerospace, automotive, transportation systems, and vehicle engineering",
        "keywords": ["aerospace", "automotive", "aircraft", "vehicle", "mechanical", "aviation", "cars",
                    "transportation", "flight", "engines", "aerodynamics", "spacecraft", "automobile"],
        "target_careers": 15
    },
    
    # ==================== SCIENCE & RESEARCH (4 domains) ====================
    "life_sciences": {
        "name": "Life Sciences & Biology",
        "description": "Biology, genetics, microbiology, ecology, and life sciences research",
        "keywords": ["biology", "genetics", "microbiology", "ecology", "biotechnology", "life sciences",
                    "molecular biology", "cell biology", "biochemistry", "lab research", "organisms"],
        "target_careers": 18
    },
    
    "physical_sciences": {
        "name": "Physical Sciences",
        "description": "Physics, chemistry, astronomy, geology, and physical sciences research",
        "keywords": ["physics", "chemistry", "astronomy", "geology", "quantum", "particle physics",
                    "astrophysics", "materials", "research", "laboratory", "experiments", "science"],
        "target_careers": 15
    },
    
    "social_sciences": {
        "name": "Social Sciences",
        "description": "Sociology, psychology, anthropology, economics, and social research",
        "keywords": ["sociology", "anthropology", "economics", "social research", "human behavior",
                    "social sciences", "political science", "demographics", "social studies", "research"],
        "target_careers": 15
    },
    
    "environmental_sustainability": {
        "name": "Environmental & Sustainability",
        "description": "Environmental science, conservation, sustainability, and climate science",
        "keywords": ["environmental", "sustainability", "conservation", "climate", "ecology", "green energy",
                    "renewable", "environmental science", "climate change", "carbon", "recycling", "nature"],
        "target_careers": 15
    },
    
    # ==================== SERVICE & HOSPITALITY (5 domains) ====================
    "food_beverage": {
        "name": "Food & Beverage",
        "description": "Culinary arts, food service, restaurant management, and beverage industry",
        "keywords": ["culinary", "chef", "cooking", "food", "restaurant", "kitchen", "cuisine", "baking",
                    "pastry", "food service", "catering", "beverage", "hospitality", "menu"],
        "target_careers": 18
    },
    
    "hospitality_tourism": {
        "name": "Hospitality & Tourism",
        "description": "Hotels, tourism, event management, and guest services",
        "keywords": ["hospitality", "tourism", "hotel", "travel", "event management", "guest services",
                    "concierge", "resort", "conference", "event planning", "destination", "lodging"],
        "target_careers": 15
    },
    
    "retail_ecommerce": {
        "name": "Retail & E-commerce",
        "description": "Retail management, merchandising, e-commerce, and customer service",
        "keywords": ["retail", "ecommerce", "merchandising", "store", "sales", "customer service",
                    "inventory", "online shopping", "shopify", "amazon", "retail management", "pos"],
        "target_careers": 15
    },
    
    "transportation_logistics": {
        "name": "Transportation & Logistics",
        "description": "Supply chain, logistics, warehousing, shipping, and transportation management",
        "keywords": ["logistics", "supply chain", "transportation", "warehousing", "shipping", "freight",
                    "distribution", "inventory", "procurement", "delivery", "fleet management", "warehouse"],
        "target_careers": 15
    },
    
    "personal_services": {
        "name": "Personal Services",
        "description": "Beauty, salon, spa, personal care, and lifestyle services",
        "keywords": ["beauty", "salon", "spa", "hairstylist", "cosmetology", "personal care", "aesthetics",
                    "makeup", "skincare", "barbering", "nail technician", "wellness", "grooming"],
        "target_careers": 12
    },
    
    # ==================== LEGAL & GOVERNANCE (3 domains) ====================
    "legal_services": {
        "name": "Legal Services",
        "description": "Law, legal practice, paralegal services, and legal administration",
        "keywords": ["law", "lawyer", "attorney", "legal", "paralegal", "litigation", "contracts",
                    "legal counsel", "court", "legal services", "bar", "jurisprudence", "advocate"],
        "target_careers": 18
    },
    
    "public_policy_government": {
        "name": "Public Policy & Government",
        "description": "Government administration, public policy, civil service, and political work",
        "keywords": ["government", "public policy", "civil service", "administration", "political",
                    "policy analyst", "public sector", "federal", "state", "municipal", "governance"],
        "target_careers": 15
    },
    
    "nonprofit_social_services": {
        "name": "Non-profit & Social Services",
        "description": "Social work, community services, non-profit management, and advocacy",
        "keywords": ["social work", "nonprofit", "community services", "charity", "advocacy", "social services",
                    "community development", "ngo", "humanitarian", "case management", "outreach"],
        "target_careers": 15
    },
    
    # ==================== SKILLED TRADES (4 domains) ====================
    "construction_trades": {
        "name": "Construction Trades",
        "description": "Carpentry, plumbing, electrical work, masonry, and construction crafts",
        "keywords": ["construction", "carpentry", "plumbing", "electrician", "hvac", "masonry", "welding",
                    "contractor", "builder", "tradesman", "building trades", "renovation", "roofing"],
        "target_careers": 18
    },
    
    "automotive_repair": {
        "name": "Automotive & Repair Services",
        "description": "Auto mechanics, vehicle repair, diesel mechanics, and automotive services",
        "keywords": ["automotive", "mechanic", "auto repair", "vehicle", "diesel", "technician", "maintenance",
                    "cars", "engine repair", "diagnostics", "automobile", "garage", "transmission"],
        "target_careers": 12
    },
    
    "manufacturing_production": {
        "name": "Manufacturing & Production",
        "description": "Factory work, production operations, quality control, and manufacturing trades",
        "keywords": ["manufacturing", "production", "factory", "assembly", "quality control", "machinist",
                    "cnc", "operator", "industrial", "fabrication", "production line", "manufacturing"],
        "target_careers": 15
    },
    
    "agriculture_farming": {
        "name": "Agriculture & Farming",
        "description": "Farming, agriculture, horticulture, livestock, and agribusiness",
        "keywords": ["agriculture", "farming", "crops", "livestock", "horticulture", "agribusiness",
                    "farm management", "agricultural", "ranching", "harvest", "soil", "cultivation"],
        "target_careers": 12
    }
}

# Summary statistics
def get_taxonomy_stats():
    """Get statistics about the domain taxonomy"""
    total_domains = len(DOMAIN_TAXONOMY)
    total_target_careers = sum(domain["target_careers"] for domain in DOMAIN_TAXONOMY.values())
    
    categories = {
        "Technology": 7,
        "Healthcare & Wellness": 6,
        "Business & Finance": 6,
        "Creative & Media": 5,
        "Education & Training": 3,
        "Engineering & Manufacturing": 5,
        "Science & Research": 4,
        "Service & Hospitality": 5,
        "Legal & Governance": 3,
        "Skilled Trades": 4
    }
    
    return {
        "total_domains": total_domains,
        "total_target_careers": total_target_careers,
        "categories": categories,
        "domains": list(DOMAIN_TAXONOMY.keys())
    }

if __name__ == "__main__":
    stats = get_taxonomy_stats()
    print("=" * 60)
    print("DOMAIN TAXONOMY STATISTICS")
    print("=" * 60)
    print(f"Total Domains: {stats['total_domains']}")
    print(f"Target Total Careers: {stats['total_target_careers']}")
    print("\nDomains by Category:")
    for category, count in stats['categories'].items():
        print(f"  {category}: {count} domains")
    print("\nAll Domain IDs:")
    for i, domain_id in enumerate(stats['domains'], 1):
        target = DOMAIN_TAXONOMY[domain_id]['target_careers']
        print(f"  {i:2d}. {domain_id:30s} ({target} careers)")
