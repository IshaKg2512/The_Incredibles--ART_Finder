import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

from streamlit_option_menu import option_menu

# ---------------------------
#    Constants & Configuration
# ---------------------------

SUPPORTED_SOURCES = {
    'Google': {'icon': '🔍', 'enabled': True},
    'YouTube': {'icon': '▶️', 'enabled': True},
    'Reddit': {'icon': '📱', 'enabled': True},
    'Quora': {'icon': '❓', 'enabled': True},
    'App Reviews': {'icon': '⭐', 'enabled': True}
}

DEFAULT_COMPETITORS = ["competitor1.com", "competitor2.com"]

# ---------------------------
#    Enhanced Helper Functions
# ---------------------------

def scrape_data_source(source: str, topic: str) -> dict:
    """
    Placeholder function that *could* be used for real scraping in the future.
    """
    triggers = [f"Trigger {i} from {source}" for i in range(np.random.randint(1, 3))]
    pain_points = [f"Pain point {i} from {source}" for i in range(np.random.randint(1, 3))]
    competitor_ads = [f"Competitor ad mention {i} from {source}" for i in range(np.random.randint(0, 2))]

    # Simulate text data for sentiment analysis
    text_samples = [
        f"This is a sample text about {topic} from {source}.",
        f"Another mention of {topic} with a random sentiment from {source}."
    ]
    sentiment_scores = [TextBlob(txt).sentiment.polarity for txt in text_samples]

    return {
        'triggers': triggers,
        'pain_points': pain_points,
        'competitor_ads': competitor_ads,
        'sentiment_data': sentiment_scores
    }

def analyze_competitor_content(content: str) -> dict:
    """
    Analyzes competitor content to identify hooks, CTAs, and basic sentiment.
    """
    sentiment = TextBlob(content).sentiment
    return {
        'hooks': [f"Sample Hook from: {content[:30]}..."],
        'ctas': [f"Sample CTA from: {content[:30]}..."],
        'sentiment': sentiment.polarity
    }

def generate_word_cloud(text_data: str):
    """
    Generates a word cloud from text data.
    """
    return WordCloud(width=800, height=400, background_color='white').generate(text_data)

def gather_data_from_selected_sources(topic: str, sources_selected: Dict[str, bool]) -> pd.DataFrame:
    """
    Aggregates scraping and analysis for all enabled sources.
    Returns a DataFrame for further visualization or metric calculations.
    """
    records = []
    current_date = datetime.now()
    
    for source_name, enabled in sources_selected.items():
        if enabled:  # If user checked this source
            scraped = scrape_data_source(source_name, topic)
            record = {
                'Date': current_date,
                'Source': source_name,
                'Triggers Found': len(scraped['triggers']),
                'Competitor Ads Mentioned': len(scraped['competitor_ads']),
                'User Sentiment (avg)': np.mean(scraped['sentiment_data']) if scraped['sentiment_data'] else 0,
                'totalUsers': np.random.randint(100, 1000)  # placeholder for user data
            }
            records.append(record)
    
    return pd.DataFrame(records)

def perform_competitor_analysis(competitors: List[str]) -> pd.DataFrame:
    """
    Example of analyzing competitor websites or ads. Currently, it's just mock data.
    """
    competitor_records = []
    current_date = datetime.now()
    
    for competitor in competitors:
        sample_content = f"Some competitor content from {competitor} discussing X, Y, Z."
        analysis = analyze_competitor_content(sample_content)
        
        competitor_records.append({
            'Date': current_date,
            'Competitor': competitor,
            'Sample Hook': analysis['hooks'][0],
            'Sample CTA': analysis['ctas'][0],
            'Sentiment Score': analysis['sentiment']
        })
    
    return pd.DataFrame(competitor_records)

# ---------------------------------
#    Content Insights Generator
# ---------------------------------

def generate_content_insights(df: pd.DataFrame) -> dict:
    """
    Generates example hooks, CTAs, and angles from a given DataFrame.
    """
    if df.empty:
        return {"hooks": [], "ctas": [], "angles": []}
    
    avg_sentiment = df["User Sentiment (avg)"].mean()
    triggers_sum = df["Triggers Found"].sum()
    
    hooks = [
        f"Overcome {triggers_sum} hurdles with our solution!",
        "Transform user doubts into brand loyalty now!"
    ]
    ctas = [
        "Try our free demo today!",
        "Join the movement and see instant results."
    ]
    
    if avg_sentiment < 0:
        angles = [
            "Address negative perceptions by offering transparent pricing.",
            "Highlight social proof to counter skeptical audiences."
        ]
    else:
        angles = [
            "Leverage positive buzz to build brand ambassadors.",
            "Celebrate success stories and user testimonials."
        ]
    
    return {"hooks": hooks, "ctas": ctas, "angles": angles}

# ---------------------------
#    Research Dashboard Components
# ---------------------------

def display_research_metrics(df: pd.DataFrame):
    """
    Displays key research metrics in a dashboard format.
    """
    st.markdown("### 📊 Research Analytics")
    
    if df.empty:
        st.warning("No data available. Please run a search first.")
        return

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_triggers = df['Triggers Found'].sum()
        st.metric("Total Triggers", f"{total_triggers:,}")
        
    with col2:
        avg_sentiment = df['User Sentiment (avg)'].mean()
        st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
        
    with col3:
        total_competitors = df['Competitor Ads Mentioned'].sum()
        st.metric("Competitor Mentions", f"{total_competitors:,}")
        
    with col4:
        if 'totalUsers' in df.columns:
            engagement_rate = (df['Triggers Found'] / df['totalUsers']).mean() * 100
            st.metric("Engagement Rate", f"{engagement_rate:.1f}%")
        else:
            st.metric("Engagement Rate", "N/A")

def display_competitor_analysis_chart(df: pd.DataFrame):
    """
    Shows competitor mentions over time in a line chart.
    """
    st.markdown("### 🎯 Competitor Strategy Analysis")
    
    if df.empty:
        st.info("No competitor data available. Run the research to see results.")
        return
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Competitor Ads Mentioned'],
        mode='lines+markers',
        name='Ad Mentions'
    ))
    fig.update_layout(title="Competitor Ads Mentioned Over Time")
    st.plotly_chart(fig, use_container_width=True)

def display_pain_points_analysis(text_data: str):
    """
    Visualizes pain points with a word cloud.
    """
    st.markdown("### 😟 Pain Points Analysis")
    if not text_data:
        st.info("No text data available for word cloud.")
        return
    
    wordcloud = generate_word_cloud(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    # Use st.pyplot() with no arguments to render the current figure
    st.pyplot()

def display_competitor_content_analysis(df_comp: pd.DataFrame):
    """
    Shows competitor hooks, CTAs, sentiment in a table and bar chart.
    """
    st.markdown("### 🏆 Competitor Content Insights")
    if df_comp.empty:
        st.info("No competitor content to analyze.")
        return
    
    st.dataframe(df_comp)

    fig = px.bar(
        df_comp, 
        x="Competitor", 
        y="Sentiment Score", 
        color="Competitor",
        title="Competitor Sentiment Scores"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
#    Fallback: Mock Data for Testing
# ---------------------------

def simulate_research_data(period='last_30_days', topic=None, competitors=None):
    """
    Generates mock data for demonstration, with optional topic & competitor info.
    """
    current_date = datetime.now()
    
    if period == 'last_30_days':
        days = 30
    elif period == 'last_90_days':
        days = 90
    else:
        days = 7
        
    dates = pd.date_range(end=current_date, periods=days)
    
    data = []
    sources = ['Google', 'YouTube', 'Reddit', 'Quora', 'App Reviews']
    for date in dates:
        for source in sources:
            data.append({
                'Date': date,
                'Source': source,
                'Triggers Found': np.random.randint(1, 20),
                'Competitor Ads Mentioned': np.random.randint(0, 5),
                'User Sentiment (avg)': np.random.uniform(-1, 1),
                'totalUsers': np.random.randint(100, 1000)
            })
    
    df = pd.DataFrame(data)
    # Just store topic and competitor list (as a string) to show they're not ignored.
    df['Topic'] = topic if topic else "N/A"
    df['Competitors'] = ", ".join(competitors) if competitors else "N/A"

    return df

# ---------------------------
#    Main App Logic
# ---------------------------

def main():
    st.set_page_config(page_title="ART Finder", layout="wide")
    
    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="ART Finder Tools",
            options=["Research Dashboard", "Competitor Analysis", "Content Insights", "Settings"],
            icons=['search', 'graph-up', 'lightbulb', 'gear'],
            menu_icon="tools",
            default_index=0
        )
        
        # Data source selection
        st.markdown("### Data Sources")
        for source, details in SUPPORTED_SOURCES.items():
            details['enabled'] = st.checkbox(
                f"{details['icon']} {source}",
                value=details['enabled']
            )
    
    # ---------------------------------
    # 1) RESEARCH DASHBOARD PAGE
    # ---------------------------------
    if selected == "Research Dashboard":
        st.title("🔍 ART Finder Dashboard")
        st.markdown("Automated Research & Trigger Analysis")

        # If no submission yet, show a quick explanation
        st.write("Use the form below to specify your research topic, timeframe, and competitors. Then click **Start Research**.")

        with st.form("research_form"):
            topic = st.text_input("Research Topic or Brand (e.g., 'Fitness Tracker')")
            
            col1, col2 = st.columns(2)
            with col1:
                timeframe = st.selectbox(
                    "Analysis Timeframe",
                    ["Last 7 days", "Last 30 days", "Last 90 days"]
                )
            with col2:
                competitor_list = st.multiselect(
                    "Competitor Websites",
                    DEFAULT_COMPETITORS
                )
            
            submitted = st.form_submit_button("Start Research")
            
            if submitted:
                with st.spinner("Gathering insights..."):
                    # Option A: Real scraping (uncomment if you want to integrate actual scraping)
                    # df = gather_data_from_selected_sources(
                    #     topic, {src: details['enabled'] for src, details in SUPPORTED_SOURCES.items()}
                    # )
                    
                    # Option B: Simulated data
                    df = simulate_research_data(
                        period=timeframe.lower().replace(" ", "_"),
                        topic=topic,
                        competitors=competitor_list
                    )

                    # Store DataFrame in session
                    st.session_state['research_data'] = df

                    # Display results
                    display_research_metrics(df)
                    if not df.empty:
                        display_competitor_analysis_chart(df)
                    
                    sample_text_data = " ".join([
                        f"{row['Source']} trigger painpoint" for _, row in df.iterrows()
                    ])
                    display_pain_points_analysis(sample_text_data)

                    # Export
                    st.download_button(
                        "📥 Export Research Report",
                        data=df.to_csv(index=False),
                        file_name="art_finder_report.csv",
                        mime="text/csv"
                    )
    
    # ---------------------------------
    # 2) COMPETITOR ANALYSIS PAGE
    # ---------------------------------
    elif selected == "Competitor Analysis":
        st.title("🏅 Competitor Analysis")
        st.markdown("Deep-dive into competitor hooks, CTAs, and sentiment.")

        # Show a placeholder explanation before the button is clicked
        st.write("Select one or more competitor websites below and click **Analyze Competitors** to see insights.")

        competitors = st.multiselect("Select Competitors", DEFAULT_COMPETITORS, DEFAULT_COMPETITORS)
        
        if st.button("Analyze Competitors"):
            with st.spinner("Analyzing competitor content..."):
                df_comp = perform_competitor_analysis(competitors)
                display_competitor_content_analysis(df_comp)
        else:
            st.info("No competitor analysis has been run yet. Select competitors and click the button above.")

    # ---------------------------------
    # 3) CONTENT INSIGHTS PAGE
    # ---------------------------------
    elif selected == "Content Insights":
        st.title("💡 Content Insights")
        st.markdown("Turn your research data into actionable hooks, CTAs, and angles.")

        # Check session_state for research data
        if 'research_data' not in st.session_state:
            st.warning("No research data found. Please run the 'Research Dashboard' first.")
        else:
            df = st.session_state['research_data']
            if df.empty:
                st.warning("Your research dataset is empty. Please run the 'Research Dashboard' with valid inputs.")
            else:
                st.write("Below is a quick summary of your recent research data:")
                st.dataframe(df.head(10))

                if st.button("Generate Content Insights"):
                    insights = generate_content_insights(df)
                    if not any(insights.values()):
                        st.info("No insights found. Please check your data.")
                    else:
                        st.subheader("Suggested Hooks")
                        for hook in insights["hooks"]:
                            st.write(f"- {hook}")
                        
                        st.subheader("Suggested CTAs")
                        for cta in insights["ctas"]:
                            st.write(f"- {cta}")
                        
                        st.subheader("Suggested Angles")
                        for angle in insights["angles"]:
                            st.write(f"- {angle}")
                else:
                    st.info("Click **Generate Content Insights** to see hooks, CTAs, and angles based on your data.")

    # ---------------------------------
    # 4) SETTINGS PAGE
    # ---------------------------------
    else:  # "Settings"
        st.title("⚙️ Settings")
        st.markdown("Configure API keys, scraping intervals, and other system preferences here.")
        
        st.write("""
        **Under Construction**:
        - You can add fields for your API keys (e.g., YouTube, Reddit).
        - Specify how often you'd like to scrape or refresh data.
        - Adjust thresholds for sentiment or competitor analysis.

        This page is just a placeholder for now.
        """)

if __name__ == "__main__":
    main()
