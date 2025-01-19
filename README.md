# AdWise
AdVise AI is an intelligent ad research platform that transforms your advertising vision into data-backed strategies. By analyzing cross-platform performance data and user behavior, we help you craft ads that resonate with your target audience and align with your brand identity.


# Check out our project & demo video here:
yy -website
yy - demo video

## Features

* **Smart Input Analysis**
    - Brand Guidelines Integration
    - Ad Style Preference Selection (Professional, Fun, Minimalist, etc.)
    - Target Platform Optimization
    - Campaign Objective Mapping

* **Cross-Platform Intelligence**
    - Platform-specific Performance Metrics
    - Audience Behavior Analysis
    - Content Format Recommendations
    - Engagement Pattern Recognition

* **AI-Powered Insights**
    - Pain Point Identification
    - Emotional Trigger Analysis
    - Interactive Hook Word Cloud
    - High-Converting CTA Recommendations

* **Performance Analytics**
    - Platform Success Metrics
    - A/B Testing Suggestions
    - Engagement Rate Benchmarking
    - ROI Optimization Tips

## Technologies Used
- LangFlow: Powers our natural language processing and insight generation
- DataStax: Enables real-time data processing and analysis
- Streamlit: Delivers an intuitive and responsive user interface
- Python 3.8+: Provides the foundation for our analytics engine

## How It Works

1. **Input Collection**
   - Define your product/service category
   - Select desired ad style
   - Input brand guidelines
   - Choose target platforms

2. **Analysis & Processing**
   - Analyze successful ads in your category
   - Process user engagement patterns
   - Generate performance metrics
   - Identify winning elements

3. **Insight Generation**
   - Create interactive hook word clouds
   - Map platform-specific performance data
   - Generate customized recommendations
   - Provide optimization suggestions

## Output Deliverables
- Platform-specific performance metrics
- Interactive hook word cloud visualization
- Identified user pain points and triggers
- Customized CTA recommendations
- Ad improvement suggestions
- Cross-platform optimization tips

## Setup Instructions

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/social-media-analytics.git
cd social-media-analytics
```

### 2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with your API keys and configuration
```

## Usage

### 1. Start the Streamlit application:
```bash
streamlit run app.py
```

### 2. Navigate to `http://localhost:8501` in your web browser

### 3. Connect your social media accounts through the settings panel

### 4. Start analyzing your social media performance!

## Configuration

Configure your application by setting the following environment variables in `.env`:

```
DATASTAX_API_KEY=your_datastax_api_key
LANGFLOW_API_KEY=your_langflow_api_key
```

## Project Structure
- main.py: Main application file containing the Streamlit app logic.
- requirements.txt: List of dependencies required for the project.

## Acknowledgments

- DataStax team for their excellent database solution
- LangFlow community for AI analytics capabilities
- Streamlit team for the amazing web framework
