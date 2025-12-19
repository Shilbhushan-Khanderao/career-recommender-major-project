"""
Massive Training Dataset Generator
Generates 200K+ diverse training samples across 48 domains
Target: ~5,000 samples per domain with high diversity
"""

import csv
import random
from pathlib import Path
import sys

# Add parent directory to import domain taxonomy
sys.path.append(str(Path(__file__).parent))
from domain_taxonomy import DOMAIN_TAXONOMY

# Expanded expression templates for maximum diversity (50+ templates)
EXPRESSION_TEMPLATES = [
    "I enjoy {activity}",
    "I'm passionate about {keyword}",
    "I love working with {keyword}",
    "I want to work in {keyword}",
    "I'm interested in {keyword} and {keyword}",
    "I have experience with {keyword}",
    "I'd like to learn more about {keyword}",
    "{keyword} fascinates me",
    "I'm skilled at {keyword}",
    "My dream is to work in {keyword}",
    "I want to pursue a career in {keyword}",
    "I'm looking for opportunities in {keyword}",
    "I excel at {keyword}",
    "I have a strong background in {keyword}",
    "I'm drawn to {keyword}",
    "{keyword} is my passion",
    "I want to specialize in {keyword}",
    "I'm eager to explore {keyword}",
    "I find {keyword} very rewarding",
    "I'm committed to {keyword}",
    "I thrive when working on {keyword}",
    "My goal is to become proficient in {keyword}",
    "I enjoy the challenge of {keyword}",
    "I'm motivated by {keyword}",
    "I have a talent for {keyword}",
    "{keyword} excites me professionally",
    "I want to build a career around {keyword}",
    "I'm naturally inclined towards {keyword}",
    "I find fulfillment in {keyword}",
    "My strengths lie in {keyword}",
    "I aspire to work with {keyword}",
    "I'm dedicated to mastering {keyword}",
    "{keyword} aligns with my career goals",
    "I have deep interest in {keyword}",
    "I want to contribute to {keyword}",
    "I'm enthusiastic about {keyword}",
    "{keyword} drives my professional ambitions",
    "I see my future in {keyword}",
    "I'm particularly good at {keyword}",
    "{keyword} is where I want to focus my career",
    "I'm always learning about {keyword}",
    "I want to make an impact in {keyword}",
    "I value working on {keyword}",
    "{keyword} matches my skills perfectly",
    "I'm seeking growth in {keyword}",
    "I want to innovate in {keyword}",
    "{keyword} is my area of expertise",
    "I'm building my career in {keyword}",
    "My experience includes {keyword}",
    "I want to advance in {keyword}",
]

# Compound sentence templates for complexity
COMPOUND_TEMPLATES = [
    "I'm interested in {keyword} and I also enjoy {activity}",
    "I have skills in {keyword}, and I want to develop expertise in {keyword}",
    "I love {keyword} because it combines {keyword} with {keyword}",
    "My background is in {keyword}, but I'm also passionate about {keyword}",
    "I excel at {keyword} and I'm always learning about {keyword}",
    "I'm fascinated by {keyword} and how it relates to {keyword}",
    "I want to work with {keyword} while also focusing on {keyword}",
    "I enjoy {activity} and I'm particularly interested in {keyword}",
    "I have experience with {keyword}, which I use for {keyword}",
    "I'm passionate about {keyword} and I want to specialize in {keyword}",
    "I like {keyword} because it involves {keyword} and {keyword}",
    "I'm skilled in {keyword} and I'm eager to apply it to {keyword}",
    "I find {keyword} rewarding, especially when combined with {keyword}",
    "My goal is to master {keyword} and contribute to {keyword}",
    "I enjoy the intersection of {keyword} and {keyword}",
    "I'm drawn to {keyword} and I want to explore {keyword}",
    "I have talent in {keyword} and I'm developing skills in {keyword}",
    "I love {activity} and I want to build a career in {keyword}",
    "I'm enthusiastic about {keyword} and I see potential in {keyword}",
    "I want to combine my interest in {keyword} with {keyword}",
]

# Multi-interest templates for diverse expressions
MULTI_INTEREST_TEMPLATES = [
    "I'm interested in {keyword}, {keyword}, and {keyword}",
    "My passions include {keyword}, {keyword}, and {activity}",
    "I enjoy working with {keyword}, {keyword}, and {keyword}",
    "I want to develop skills in {keyword}, {keyword}, and {keyword}",
    "I have experience in {keyword}, {keyword}, and {keyword}",
    "I'm fascinated by {keyword}, {keyword}, and {keyword}",
    "My career interests span {keyword}, {keyword}, and {keyword}",
    "I excel at {keyword}, {keyword}, and {keyword}",
    "I'm passionate about {keyword}, {keyword}, and {activity}",
    "I want to work with {keyword}, {keyword}, and {keyword}",
]

# Question-style templates
QUESTION_TEMPLATES = [
    "How can I work in {keyword}?",
    "What careers involve {keyword}?",
    "Is {keyword} a good career path?",
    "How do I start in {keyword}?",
    "What skills do I need for {keyword}?",
    "Can I build a career in {keyword}?",
    "What jobs focus on {keyword}?",
    "How do I get into {keyword}?",
    "What are the opportunities in {keyword}?",
    "Should I pursue {keyword}?",
]

# Descriptive statement templates
DESCRIPTIVE_TEMPLATES = [
    "I have {number} years of experience in {keyword}",
    "I recently graduated with a degree in {keyword}",
    "I'm currently studying {keyword}",
    "I've completed certifications in {keyword}",
    "I've been working in {keyword} industry",
    "I'm transitioning my career to {keyword}",
    "I've developed strong skills in {keyword}",
    "I'm actively involved in {keyword} projects",
    "I have professional experience with {keyword}",
    "I'm seeking to advance my career in {keyword}",
]

def generate_samples_for_domain(domain_id, target_count=5000):
    """Generate diverse training samples for a specific domain"""
    domain_info = DOMAIN_TAXONOMY[domain_id]
    keywords = domain_info["keywords"]
    
    # Expanded activities specific to domain types
    base_activities = [
        f"working on {keywords[0]} projects",
        f"solving {keywords[1]} problems",
        f"developing {keywords[2]} solutions",
        f"analyzing {keywords[0]} data",
        f"creating {keywords[1]} systems",
        f"building {keywords[2]} applications",
        f"designing {keywords[0]} architectures",
        f"implementing {keywords[1]} strategies",
        f"researching {keywords[2]} technologies",
        f"collaborating on {keywords[0]} initiatives"
    ]
    
    # Generate additional activities from keywords
    extended_activities = base_activities + [
        f"{keyword} development" for keyword in keywords[:10]
    ] + [
        f"{keyword} implementation" for keyword in keywords[:8]
    ]
    
    samples = []
    templates_pool = (EXPRESSION_TEMPLATES + COMPOUND_TEMPLATES + 
                     MULTI_INTEREST_TEMPLATES + QUESTION_TEMPLATES + 
                     DESCRIPTIVE_TEMPLATES)
    
    print(f"  Generating {target_count} samples for {domain_id}...")
    
    for i in range(target_count):
        # Select template
        if i % 100 < 40:  # 40% simple expressions
            template = random.choice(EXPRESSION_TEMPLATES)
        elif i % 100 < 70:  # 30% compound sentences
            template = random.choice(COMPOUND_TEMPLATES)
        elif i % 100 < 85:  # 15% multi-interest
            template = random.choice(MULTI_INTEREST_TEMPLATES)
        elif i % 100 < 95:  # 10% questions
            template = random.choice(QUESTION_TEMPLATES)
        else:  # 5% descriptive statements
            template = random.choice(DESCRIPTIVE_TEMPLATES)
        
        # Fill in template
        text = template
        
        # Replace {keyword} placeholders
        while "{keyword}" in text:
            keyword = random.choice(keywords)
            text = text.replace("{keyword}", keyword, 1)
        
        # Replace {activity} placeholders
        while "{activity}" in text:
            activity = random.choice(extended_activities)
            text = text.replace("{activity}", activity, 1)
        
        # Replace {number} placeholders (for experience years)
        while "{number}" in text:
            number = random.choice(["2", "3", "5", "7", "10", "15"])
            text = text.replace("{number}", number, 1)
        
        # Clean up text
        text = text.strip()
        
        # Add variation in capitalization (10% all lowercase for natural queries)
        if random.random() < 0.1:
            text = text.lower()
        
        samples.append({"text": text, "label": domain_id})
    
    return samples

def deduplicate_samples(samples):
    """Remove duplicate samples while preserving domain balance"""
    print("\nDeduplicating samples...")
    
    unique_texts = set()
    deduplicated = []
    duplicates_count = 0
    
    for sample in samples:
        if sample["text"] not in unique_texts:
            unique_texts.add(sample["text"])
            deduplicated.append(sample)
        else:
            duplicates_count += 1
    
    print(f"  Removed {duplicates_count:,} duplicates")
    print(f"  Kept {len(deduplicated):,} unique samples")
    
    return deduplicated

def generate_massive_dataset(target_total=250000):
    """Generate massive training dataset across all domains"""
    
    total_domains = len(DOMAIN_TAXONOMY)
    samples_per_domain = target_total // total_domains
    
    print("=" * 70)
    print("MASSIVE TRAINING DATASET GENERATOR")
    print("=" * 70)
    print(f"Target: {target_total:,} samples across {total_domains} domains")
    print(f"Samples per domain: ~{samples_per_domain:,}\n")
    
    all_samples = []
    
    for domain_id in DOMAIN_TAXONOMY.keys():
        samples = generate_samples_for_domain(domain_id, samples_per_domain)
        all_samples.extend(samples)
    
    print(f"\nGenerated {len(all_samples):,} total samples")
    
    # Deduplicate
    all_samples = deduplicate_samples(all_samples)
    
    # Shuffle for better training
    random.shuffle(all_samples)
    
    return all_samples

def save_dataset_csv(samples, output_path):
    """Save dataset to CSV file"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"\nSaved {len(samples):,} samples to {output_path}")

def print_statistics(samples):
    """Print dataset statistics"""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    # Count by domain
    domain_counts = {}
    for sample in samples:
        domain = sample['label']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"\nSamples per domain:")
    for domain in sorted(domain_counts.keys()):
        count = domain_counts[domain]
        domain_name = DOMAIN_TAXONOMY[domain]["name"]
        print(f"  {domain:30s} : {count:>6,} samples - {domain_name}")
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(samples):,} unique training samples")
    print(f"{'='*70}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate dataset (targeting 350K to get ~200K+ after deduplication)
    samples = generate_massive_dataset(target_total=350000)
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "domain_dataset.csv"
    save_dataset_csv(samples, output_path)
    
    # Print statistics
    print_statistics(samples)
    
    print(f"\n{'='*70}")
    print(f"SUCCESS: Massive dataset generated!")
    print(f"{'='*70}")
