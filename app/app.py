"""
AI-Driven Personalized Career Guidance System

Main Streamlit application for career recommendations based on user input.
Enhanced with SBERT semantic similarity and advanced visualization.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.domain_predictor import predict_domain, predict_domain_with_probabilities, is_model_loaded
from app.recommender import recommend_careers_for_domain, explain_recommendation
from app.personality import personality_scores_from_text, personality_summary
from app.nlp_pipeline import tokenize_text


# Add custom CSS for better styling (minimal)
def inject_custom_css():
    st.markdown("""
    <style>
    /* Minimal styling - Streamlit handles most UI */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Inject custom CSS
    inject_custom_css()
    
    # Page configuration
    st.set_page_config(
        page_title="AI Career Guidance System",
        page_icon="",
        layout="wide"
    )
    
    # Title and description
    st.markdown('# 🧭 AI-Driven Personalized Career Guidance System')
    st.markdown("""
    Welcome! This system uses **Machine Learning**, **Natural Language Processing**, and 
    **Semantic Embeddings** to recommend personalized career paths based on your interests, 
    skills, and preferences.
    
    Simply describe yourself in the text box below, and we'll analyze your input to suggest 
    the most suitable careers for you.
    """)
    
    # System info in sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ System Information</div>', unsafe_allow_html=True)
        st.markdown("""
        **🔧 Technologies:**
        - Domain Classification (99.30% accuracy)
        - TF-IDF + Logistic Regression
        - SBERT Semantic Similarity
        - Big Five Personality Analysis
        - Hybrid Recommendation Engine
        
        **📊 Database:**
        - 48 Career Domains
        - 767 Career Profiles
        - 187,405 Training Samples
        """)
        
        st.markdown("---")
        
        st.markdown('<div class="sidebar-header"><i class="fas fa-sliders-h"></i> Model Settings</div>', unsafe_allow_html=True)
        
        similarity_model = st.radio(
            "Similarity Engine:",
            ["TF-IDF (Classical)", "SBERT (Semantic)", "Hybrid (Both)"],
            index=2,
            help="Choose the similarity computation method"
        )
        
        st.markdown("⚡ **TF-IDF:** Fast, keyword-based matching")
        st.markdown("🧠 **SBERT:** Deep semantic understanding")
        st.markdown("🔀 **Hybrid:** Best of both approaches")
        
        st.markdown("---")
        
        show_advanced = st.checkbox("Show Advanced Analytics", value=False)
    
    # Check if model is loaded
    if not is_model_loaded():
        st.error("""
        ⚠️ **Models not found!**
        
        Please train the domain classifier first by running:
        ```
        python train/train_domain_classifier.py
        ```
        """)
        return
    
    st.markdown("---")
    
    # User input section
    st.markdown('## 📝 Tell us about yourself')
    
    user_input = st.text_area(
        "Describe your interests, skills, hobbies, and what you enjoy doing:",
        height=150,
        placeholder="Example: I love working with data and statistics. I enjoy programming in Python and building machine learning models. I'm curious about AI and want to solve complex analytical problems...",
        help="Be as detailed as possible. Mention your skills, interests, personality traits, and preferences."
    )
    
    # Advanced options (collapsible)
    with st.expander("⚙️ Display Options", expanded=False):
        top_k = st.slider(
            "Number of recommendations to show:",
            min_value=3,
            max_value=15,
            value=5,
            help="Select how many career recommendations you want to see"
        )
        
        show_scores = st.checkbox(
            "Show detailed scoring breakdown",
            value=True,
            help="Display individual scores (similarity, keywords, personality, skills)"
        )
        
        show_personality = st.checkbox(
            "Show personality analysis",
            value=True,
            help="Display detected personality traits from your input"
        )
        
        show_visualizations = st.checkbox(
            "Show interactive visualizations",
            value=True,
            help="Display charts and graphs for better insights"
        )
    
    # Recommendation button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        get_recommendations = st.button(
            "🚀 Get Career Recommendations",
            type="primary",
            use_container_width=True
        )
    
    # Process recommendations
    if get_recommendations:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text describing yourself before getting recommendations.")
        else:
            # Show loading spinner
            with st.spinner("🔍 Analyzing your input and finding the best career matches..."):
                try:
                    # Predict domain with probabilities
                    prediction_result = predict_domain_with_probabilities(user_input)
                    domain = prediction_result['domain']
                    confidence = prediction_result['confidence']
                    probabilities = prediction_result['probabilities']
                    
                    # Get personality analysis
                    user_tokens = tokenize_text(user_input)
                    personality_scores = personality_scores_from_text(user_tokens)
                    
                    # Determine similarity model parameter
                    if similarity_model == "TF-IDF (Classical)":
                        model_param = "tfidf"
                    elif similarity_model == "SBERT (Semantic)":
                        model_param = "sbert"
                    else:  # Hybrid
                        model_param = "hybrid"
                    
                    # Get recommendations with selected model
                    recommendations = recommend_careers_for_domain(
                        user_input,
                        domain,
                        top_k=top_k,
                        similarity_model=model_param
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.success("✅ Analysis complete! Here are your results:")
                    
                    # Domain prediction section
                    st.subheader("🎓 Predicted Career Domain")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Format domain name for display
                        domain_display = domain.replace('_', ' ').title()
                        st.metric(
                            label="Your Primary Domain",
                            value=domain_display,
                            help="This is the career domain that best matches your interests"
                        )
                    
                    with col2:
                        st.metric(
                            label="Confidence",
                            value=f"{confidence * 100:.1f}%",
                            help="How confident the model is in this prediction"
                        )
                    
                    # Show domain probabilities
                    if show_visualizations:
                        # Create bar chart for top domains
                        top_domains = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:10])
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[d.replace('_', ' ').title() for d in top_domains.keys()],
                                y=[p * 100 for p in top_domains.values()],
                                marker_color='#1f77b4',
                                text=[f"{p*100:.1f}%" for p in top_domains.values()],
                                textposition='auto'
                            )
                        ])
                        
                        fig.update_layout(
                            title="Top 10 Domain Matches",
                            xaxis_title="Career Domain",
                            yaxis_title="Match Probability (%)",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="domain_matches_chart")
                    else:
                        with st.expander("📊 See all domain probabilities", expanded=False):
                            prob_df = pd.DataFrame({
                                "Domain": [d.replace('_', ' ').title() for d in probabilities.keys()],
                                "Probability": [f"{p * 100:.1f}%" for p in probabilities.values()]
                            })
                            st.table(prob_df)
                    
                    # Personality analysis
                    if show_personality:
                        st.subheader("🧠 Personality Analysis")
                        summary = personality_summary(personality_scores)
                        st.info(f"**Detected Traits:** {summary}")
                        
                        if any(score > 0 for score in personality_scores.values()):
                            if show_visualizations:
                                # Create radar chart for personality
                                traits = list(personality_scores.keys())
                                scores = list(personality_scores.values())
                                
                                fig = go.Figure(data=go.Scatterpolar(
                                    r=scores,
                                    theta=[t.title() for t in traits],
                                    fill='toself',
                                    marker_color='#ff7f0e'
                                ))
                                
                                fig.update_layout(
                                    polar=dict(radialaxis=dict(visible=True, range=[0, max(scores) + 1])),
                                    showlegend=False,
                                    title="Big Five Personality Traits",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, key="personality_radar_chart")
                            else:
                                with st.expander("📈 Detailed personality scores", expanded=False):
                                    for trait, score in personality_scores.items():
                                        if score > 0:
                                            st.write(f"**{trait.title()}:** {score} relevant keywords detected")
                    
                    # Recommendations section
                    st.markdown("---")
                    st.subheader("💼 Recommended Careers")
                    
                    # Show model information
                    if model_param == "hybrid":
                        st.info("🔬 **Hybrid Mode:** Combining TF-IDF keyword matching + SBERT semantic understanding for best results")
                    elif model_param == "sbert":
                        st.info("🧠 **SBERT Mode:** Using deep semantic embeddings to understand meaning beyond keywords")
                    else:
                        st.info("📊 **TF-IDF Mode:** Using classical keyword-based similarity matching")
                    
                    if not recommendations:
                        st.warning(f"No careers found for domain: {domain}")
                    else:
                        # Show comparison chart if visualizations enabled
                        if show_visualizations and len(recommendations) >= 3:
                            careers_list = [rec['career'] for rec in recommendations[:5]]
                            scores_list = [rec['hybrid_score'] * 100 for rec in recommendations[:5]]
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    y=careers_list[::-1],  # Reverse for better display
                                    x=scores_list[::-1],
                                    orientation='h',
                                    marker_color='#2ca02c',
                                    text=[f"{s:.1f}%" for s in scores_list[::-1]],
                                    textposition='auto'
                                )
                            ])
                            
                            fig.update_layout(
                                title="Top Career Matches - Overall Scores",
                                xaxis_title="Match Score (%)",
                                yaxis_title="Career",
                                height=300,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="career_matches_chart")
                            st.markdown("---")
                        
                        for idx, rec in enumerate(recommendations, 1):
                            with st.container():
                                st.markdown(f"### {idx}. {rec['career']}")
                                
                                # Create columns for metadata
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write(f"**Domain:** {rec['domain'].replace('_', ' ').title()}")
                                
                                with col2:
                                    st.write(f"**Match Score:** {rec['hybrid_score']:.0%}")
                                
                                with col3:
                                    # Color-code match score
                                    if rec['hybrid_score'] >= 0.7:
                                        match_label = "🟢 Excellent"
                                    elif rec['hybrid_score'] >= 0.5:
                                        match_label = "🟡 Good"
                                    else:
                                        match_label = "🟠 Moderate"
                                    st.write(f"**Rating:** {match_label}")
                                
                                # Description
                                st.write(f"**Description:** {rec['description']}")
                                
                                # Required skills
                                st.write(f"**Key Skills:** {rec['skills']}")
                                
                                # Explanation
                                explanation = explain_recommendation(rec)
                                st.info(f"💡 {explanation}")
                                
                                # Detailed scores
                                if show_scores:
                                    with st.expander("📊 Score Breakdown", expanded=False):
                                        # Show model comparison for hybrid mode
                                        if model_param == "hybrid" and show_advanced:
                                            st.markdown("**Model Comparison:**")
                                            comp_cols = st.columns(2)
                                            with comp_cols[0]:
                                                st.metric("TF-IDF Score", f"{rec['tfidf_score']:.0%}")
                                            with comp_cols[1]:
                                                st.metric("SBERT Score", f"{rec['sbert_score']:.0%}")
                                            st.markdown("---")
                                        
                                        if show_visualizations:
                                            # Create pie chart for score components
                                            score_components = {
                                                'Similarity': rec['similarity_score'],
                                                'Keywords': rec['keyword_score'],
                                                'Personality': rec['personality_score'],
                                                'Skills': rec['skills_score']
                                            }
                                            
                                            fig = go.Figure(data=[go.Pie(
                                                labels=list(score_components.keys()),
                                                values=list(score_components.values()),
                                                hole=.3
                                            )])
                                            
                                            fig.update_layout(
                                                title="Score Component Breakdown",
                                                height=300
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True, key=f"score_breakdown_{idx}")
                                        else:
                                            score_cols = st.columns(4)
                                            
                                            with score_cols[0]:
                                                st.metric("Similarity", f"{rec['similarity_score']:.0%}")
                                            
                                            with score_cols[1]:
                                                st.metric("Keywords", f"{rec['keyword_score']:.0%}")
                                            
                                            with score_cols[2]:
                                                st.metric("Personality", f"{rec['personality_score']:.0%}")
                                            
                                            with score_cols[3]:
                                                st.metric("Skills", f"{rec['skills_score']:.0%}")
                                
                                st.markdown("---")
                
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p><b>⭐ AI-Driven Personalized Career Guidance System</b></p>
        <p>💻 Built with Streamlit, scikit-learn, SBERT & NLP</p>
        <p>📊 Features: 48 Domains • 767 Careers • 99.30% Accuracy • Semantic Similarity</p>
        <p>🎓 M.Tech Major Project | November 2025</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
