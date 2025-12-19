"""
Comprehensive Career Database Generator - Smart Generation System
Generates 767 diverse careers across 48 domains using intelligent templates
"""

import csv
import random
from pathlib import Path
import sys

# Add parent directory to import domain taxonomy
sys.path.append(str(Path(__file__).parent))
from domain_taxonomy import DOMAIN_TAXONOMY

# Career title templates and modifiers by domain type
CAREER_TEMPLATES = {
    "technical": {
        "titles": ["{prefix} Engineer", "{prefix} Developer", "{prefix} Architect", "{prefix} Specialist",
                  "{prefix} Analyst", "{prefix} Consultant", "{prefix} Manager", "Lead {prefix} Engineer",
                  "Senior {prefix} Developer", "{prefix} Administrator", "{prefix} Coordinator",
                  "{prefix} Technician", "Principal {prefix} Engineer", "{prefix} Designer"],
        "skills_common": ["Problem Solving", "Technical Communication", "Documentation", "Teamwork"],
        "personality": ["high_conscientiousness", "moderate_openness", "moderate_agreeableness"]
    },
    "creative": {
        "titles": ["{prefix} Designer", "{prefix} Artist", "{prefix} Creator", "{prefix} Director",
                  "{prefix} Producer", "{prefix} Manager", "Senior {prefix} Designer", "{prefix} Specialist",
                  "Lead {prefix} Designer", "{prefix} Coordinator", "Freelance {prefix} Artist"],
        "skills_common": ["Creativity", "Visual Communication", "Adobe Creative Suite", "Attention to Detail"],
        "personality": ["high_openness", "moderate_conscientiousness", "moderate_extraversion"]
    },
    "healthcare": {
        "titles": ["{prefix} Physician", "{prefix} Nurse", "{prefix} Therapist", "{prefix} Specialist",
                  "{prefix} Technician", "{prefix} Coordinator", "{prefix} Assistant", "{prefix} Practitioner",
                  "Lead {prefix} Nurse", "{prefix} Manager", "{prefix} Consultant"],
        "skills_common": ["Patient Care", "Medical Knowledge", "Empathy", "Communication"],
        "personality": ["high_agreeableness", "high_conscientiousness", "moderate_openness"]
    },
    "business": {
        "titles": ["{prefix} Manager", "{prefix} Analyst", "{prefix} Consultant", "{prefix} Director",
                  "{prefix} Coordinator", "{prefix} Specialist", "Senior {prefix} Manager", "{prefix} Executive",
                  "{prefix} Administrator", "{prefix} Associate", "Lead {prefix} Analyst"],
        "skills_common": ["Leadership", "Strategic Planning", "Communication", "Analysis"],
        "personality": ["high_conscientiousness", "moderate_extraversion", "moderate_openness"]
    },
    "service": {
        "titles": ["{prefix} Specialist", "{prefix} Manager", "{prefix} Coordinator", "{prefix} Supervisor",
                  "{prefix} Associate", "Senior {prefix} Specialist", "{prefix} Consultant", "{prefix} Director",
                  "{prefix} Representative", "{prefix} Agent"],
        "skills_common": ["Customer Service", "Communication", "Problem Solving", "Organization"],
        "personality": ["high_agreeableness", "high_extraversion", "moderate_conscientiousness"]
    },
    "trades": {
        "titles": ["{prefix} Technician", "{prefix} Specialist", "Master {prefix}", "{prefix} Installer",
                  "Licensed {prefix}", "{prefix} Mechanic", "{prefix} Operator", "{prefix} Supervisor",
                  "{prefix} Foreman", "Journey-level {prefix}"],
        "skills_common": ["Technical Skills", "Safety", "Manual Dexterity", "Problem Solving"],
        "personality": ["high_conscientiousness", "moderate_openness", "low_neuroticism"]
    }
}

# Map domains to category types
DOMAIN_CATEGORIES = {
    "data_science_ai": "technical",
    "software_development": "technical",
    "cybersecurity_it": "technical",
    "cloud_devops": "technical",
    "hardware_electronics": "technical",
    "gaming_xr": "technical",
    "telecommunications": "technical",
    "medical_clinical": "healthcare",
    "mental_health_therapy": "healthcare",
    "allied_health": "healthcare",
    "public_health": "healthcare",
    "pharmaceutical_biotech": "healthcare",
    "alternative_wellness": "healthcare",
    "corporate_management": "business",
    "entrepreneurship": "business",
    "finance_investment": "business",
    "accounting_audit": "business",
    "sales_business_dev": "business",
    "consulting_strategy": "business",
    "design_visual_arts": "creative",
    "writing_publishing": "creative",
    "film_video_production": "creative",
    "music_audio": "creative",
    "performing_arts": "creative",
    "academic_education": "service",
    "corporate_training": "service",
    "educational_technology": "technical",
    "mechanical_industrial": "technical",
    "civil_construction": "technical",
    "electrical_electronics_eng": "technical",
    "chemical_materials": "technical",
    "aerospace_automotive": "technical",
    "life_sciences": "technical",
    "physical_sciences": "technical",
    "social_sciences": "service",
    "environmental_sustainability": "technical",
    "food_beverage": "service",
    "hospitality_tourism": "service",
    "retail_ecommerce": "service",
    "transportation_logistics": "service",
    "personal_services": "service",
    "legal_services": "business",
    "public_policy_government": "business",
    "nonprofit_social_services": "service",
    "construction_trades": "trades",
    "automotive_repair": "trades",
    "manufacturing_production": "trades",
    "agriculture_farming": "trades"
}

def generate_career_name(domain_id, index, total):
    """Generate a realistic career name based on domain"""
    category = DOMAIN_CATEGORIES.get(domain_id, "business")
    templates = CAREER_TEMPLATES[category]["titles"]
    
    # Get domain keywords for prefixes
    keywords = DOMAIN_TAXONOMY[domain_id]["keywords"]
    domain_name = DOMAIN_TAXONOMY[domain_id]["name"]
    
    # Select diverse prefixes from keywords
    prefix_options = [k.title() for k in keywords[:15]]  # Use first 15 keywords
    
    # Add some domain-specific role prefixes
    if index < len(prefix_options):
        prefix = prefix_options[index]
    else:
        prefix = random.choice(prefix_options)
    
    # Select template
    template_index = index % len(templates)
    template = templates[template_index]
    
    # Generate career title
    career_title = template.replace("{prefix}", prefix)
    
    # Clean up awkward names
    career_title = career_title.replace("  ", " ").strip()
    
    return career_title

def generate_skills(domain_id, career_name):
    """Generate skill set based on domain and career"""
    category = DOMAIN_CATEGORIES.get(domain_id, "business")
    common_skills = CAREER_TEMPLATES[category]["skills_common"]
    
    # Get domain-specific skills from keywords
    keywords = DOMAIN_TAXONOMY[domain_id]["keywords"]
    domain_skills = [k.title() for k in random.sample(keywords, min(5, len(keywords)))]
    
    # Combine and create unique skill set
    all_skills = domain_skills + common_skills
    selected_skills = random.sample(all_skills, min(7, len(all_skills)))
    
    return ",".join(selected_skills)

def generate_personality(domain_id):
    """Generate personality traits based on domain category"""
    category = DOMAIN_CATEGORIES.get(domain_id, "business")
    base_personality = CAREER_TEMPLATES[category]["personality"]
    
    # Add some variation
    variations = ["low_neuroticism", "moderate_agreeableness", "moderate_extraversion", 
                 "moderate_conscientiousness", "moderate_openness"]
    
    personality_traits = base_personality[:2] + [random.choice(variations)]
    return ",".join(personality_traits[:2])  # Return 2 traits

def generate_keywords(domain_id, career_name):
    """Generate keywords from domain and career name"""
    keywords = DOMAIN_TAXONOMY[domain_id]["keywords"]
    
    # Sample keywords
    selected = random.sample(keywords, min(9, len(keywords)))
    
    return ",".join(selected)

def generate_description(domain_id, career_name):
    """Generate career description"""
    domain_desc = DOMAIN_TAXONOMY[domain_id]["description"]
    category = DOMAIN_CATEGORIES.get(domain_id, "business")
    
    # Description templates
    templates = [
        f"Specializes in {domain_desc.lower()} with focus on delivering high-quality results and innovative solutions.",
        f"Works in {domain_desc.lower()} field, providing expertise and professional services to clients and organizations.",
        f"Applies specialized knowledge in {domain_desc.lower()} to solve complex problems and drive organizational success.",
        f"Contributes to {domain_desc.lower()} through technical expertise, strategic thinking, and collaborative teamwork.",
        f"Delivers professional services in {domain_desc.lower()} with emphasis on quality, efficiency, and client satisfaction.",
        f"Provides expert-level support in {domain_desc.lower()} combining technical skills with strong communication abilities.",
        f"Leads initiatives in {domain_desc.lower()} ensuring best practices and continuous improvement in operations.",
        f"Develops and implements solutions in {domain_desc.lower()} while maintaining high standards of professional excellence."
    ]
    
    return random.choice(templates)

def generate_all_careers():
    """Generate all careers across all domains"""
    all_careers = []
    
    for domain_id, domain_info in DOMAIN_TAXONOMY.items():
        target_count = domain_info["target_careers"]
        
        print(f"Generating {target_count} careers for {domain_id}...")
        
        for i in range(target_count):
            career_name = generate_career_name(domain_id, i, target_count)
            
            career = {
                "career": career_name,
                "domain": domain_id,
                "skills": generate_skills(domain_id, career_name),
                "personality": generate_personality(domain_id),
                "keywords": generate_keywords(domain_id, career_name),
                "description": generate_description(domain_id, career_name)
            }
            
            all_careers.append(career)
    
    print(f"\nTotal careers generated: {len(all_careers)}")
    return all_careers

def save_careers_csv(careers, output_path):
    """Save careers to CSV file"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['career', 'domain', 'skills', 'personality', 'keywords', 'description'])
        writer.writeheader()
        writer.writerows(careers)
    
    print(f"Saved {len(careers)} careers to {output_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE CAREER DATABASE GENERATOR")
    print("=" * 70)
    print(f"Target: {sum(d['target_careers'] for d in DOMAIN_TAXONOMY.values())} careers across {len(DOMAIN_TAXONOMY)} domains\n")
    
    # Generate careers
    careers = generate_all_careers()
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "careers_dataset.csv"
    save_careers_csv(careers, output_path)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    
    domain_counts = {}
    for career in careers:
        domain = career['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"\nCareers per domain:")
    for domain, count in sorted(domain_counts.items()):
        domain_name = DOMAIN_TAXONOMY[domain]["name"]
        print(f"  {domain:30s} : {count:3d} careers - {domain_name}")
    
    print(f"\n{'='*70}")
    print(f"SUCCESS: Generated {len(careers)} diverse careers!")
    print(f"{'='*70}")
