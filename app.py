"""
Customer Support RAG with Sentiment Analysis - Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import logging
from typing import Dict, List, Any, Optional

# Import our custom modules
from src.rag_system import RAGSystem
from src.sentiment_analyzer import SentimentAnalyzer
from src.escalation_predictor import EscalationPredictor
from src.response_generator import ResponseGenerator
from src.utils import (
    load_json_file, 
    generate_customer_id, 
    clean_text, 
    get_time_ago,
    create_response_summary
)

# Configure page
st.set_page_config(
    page_title="Customer Support RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.conversation_history = []
    st.session_state.customer_id = None
    st.session_state.analytics_data = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sentiment-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .escalation-high {
        border-left-color: #dc3545 !important;
        background: #f8d7da;
    }
    
    .escalation-medium {
        border-left-color: #ffc107 !important;
        background: #fff3cd;
    }
    
    .escalation-low {
        border-left-color: #28a745 !important;
        background: #d4edda;
    }
    
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        min-width: 150px;
    }
    
    .response-suggestions {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize system components
@st.cache_resource
def initialize_system():
    """Initialize all system components"""
    try:
        # Initialize components
        rag_system = RAGSystem()
        sentiment_analyzer = SentimentAnalyzer(use_transformer=True)
        escalation_predictor = EscalationPredictor()
        response_generator = ResponseGenerator()
        
        # Load knowledge base
        knowledge_base = load_json_file("data/knowledge_base.json")
        if knowledge_base:
            rag_system.add_documents(knowledge_base)
            st.success(f"✅ Loaded {len(knowledge_base)} articles into knowledge base")
        
        # Load sample conversations for demo
        sample_conversations = load_json_file("data/sample_conversations.json")
        
        return {
            'rag_system': rag_system,
            'sentiment_analyzer': sentiment_analyzer,
            'escalation_predictor': escalation_predictor,
            'response_generator': response_generator,
            'sample_conversations': sample_conversations
        }
        
    except Exception as e:
        st.error(f"❌ Error initializing system: {e}")
        return None

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Customer Support RAG with Sentiment Analysis</h1>
        <p>Intelligent customer support with real-time sentiment analysis and escalation prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("🔄 Initializing Customer Support RAG System..."):
            system = initialize_system()
            if system:
                st.session_state.system = system
                st.session_state.initialized = True
            else:
                st.error("Failed to initialize system. Please check your configuration.")
                return
    
    system = st.session_state.system
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # Customer Information
        st.subheader("👤 Customer Information")
        customer_email = st.text_input("Customer Email", placeholder="customer@example.com")
        if customer_email:
            st.session_state.customer_id = generate_customer_id(email=customer_email)
            st.success(f"Customer ID: {st.session_state.customer_id[:8]}...")
        
        # Demo Data
        st.subheader("📚 Demo Data")
        if st.button("Load Sample Conversation"):
            if system['sample_conversations']:
                sample = system['sample_conversations'][0]  # Load first sample
                st.session_state.conversation_history = sample['messages']
                st.success("Sample conversation loaded!")
                st.rerun()
        
        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            st.success("Conversation cleared!")
            st.rerun()
        
        # System Stats
        st.subheader("📊 System Statistics")
        kb_stats = system['rag_system'].get_collection_stats()
        st.metric("Knowledge Articles", kb_stats.get('total_documents', 0))
        st.metric("Conversations Today", len(st.session_state.analytics_data))
        
        # Settings
        st.subheader("⚙️ Settings")
        show_debug = st.checkbox("Show Debug Information", value=False)
        auto_respond = st.checkbox("Auto-generate Responses", value=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Interface", "📊 Analytics Dashboard", "🔍 Knowledge Base", "🧪 Testing"])
    
    with tab1:
        chat_interface(system, auto_respond, show_debug)
    
    with tab2:
        analytics_dashboard(system)
    
    with tab3:
        knowledge_base_interface(system['rag_system'])
    
    with tab4:
        testing_interface(system)

def chat_interface(system, auto_respond, show_debug):
    """Main chat interface"""
    
    st.header("💬 Customer Support Chat")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("📜 Conversation History")
        
        for i, msg in enumerate(st.session_state.conversation_history):
            with st.container():
                col1, col2 = st.columns([1, 6])
                
                with col1:
                    if msg.get('speaker') == 'customer':
                        st.markdown("🧑‍💼 **Customer**")
                    else:
                        st.markdown("🤖 **Agent**")
                
                with col2:
                    message_text = msg.get('message', msg.get('content', ''))
                    st.write(message_text)
                    
                    # Show timestamp if available
                    if 'timestamp' in msg:
                        timestamp = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
                        st.caption(f"⏰ {get_time_ago(timestamp)}")
        
        st.divider()
    
    # Customer message input
    st.subheader("✍️ New Customer Message")
    customer_message = st.text_area(
        "Customer Message:",
        placeholder="Enter the customer's message here...",
        height=120,
        key="customer_input"
    )
    
    # Process message button
    col1, col2 = st.columns([1, 4])
    with col1:
        process_message = st.button("🔄 Process Message", type="primary")
    
    if process_message and customer_message:
        process_customer_message(system, customer_message, auto_respond, show_debug)

def process_customer_message(system, customer_message, auto_respond, show_debug):
    """Process a customer message through the entire pipeline"""
    
    start_time = time.time()
    
    with st.spinner("🔄 Processing customer message..."):
        
        # 1. Sentiment Analysis
        with st.status("🎭 Analyzing sentiment...", expanded=False) as status:
            sentiment_analysis = system['sentiment_analyzer'].analyze_sentiment(customer_message)
            status.update(label="✅ Sentiment analysis complete", state="complete")
        
        # 2. Escalation Prediction
        with st.status("⚠️ Predicting escalation risk...", expanded=False) as status:
            escalation_prediction = system['escalation_predictor'].predict_escalation(
                customer_id=st.session_state.customer_id or "anonymous",
                message=customer_message,
                conversation_history=st.session_state.conversation_history,
                sentiment_analysis=sentiment_analysis
            )
            status.update(label="✅ Escalation prediction complete", state="complete")
        
        # 3. Knowledge Retrieval
        with st.status("📚 Retrieving relevant knowledge...", expanded=False) as status:
            context, source_docs = system['rag_system'].get_context(customer_message, n_results=3)
            status.update(label="✅ Knowledge retrieval complete", state="complete")
        
        # 4. Response Generation (if enabled)
        response_data = None
        if auto_respond:
            with st.status("💬 Generating response...", expanded=False) as status:
                response_data = system['response_generator'].generate_response(
                    customer_message=customer_message,
                    sentiment_analysis=sentiment_analysis,
                    escalation_prediction=escalation_prediction,
                    knowledge_context=context,
                    source_documents=source_docs
                )
                status.update(label="✅ Response generation complete", state="complete")
    
    processing_time = time.time() - start_time
    
    # Display results
    display_analysis_results(
        customer_message, sentiment_analysis, escalation_prediction, 
        source_docs, response_data, processing_time, show_debug
    )
    
    # Add to conversation history
    new_message = {
        'timestamp': datetime.now().isoformat(),
        'speaker': 'customer',
        'message': customer_message,
        'sentiment_analysis': sentiment_analysis,
        'escalation_prediction': escalation_prediction
    }
    st.session_state.conversation_history.append(new_message)
    
    # Add agent response if generated
    if response_data:
        agent_message = {
            'timestamp': datetime.now().isoformat(),
            'speaker': 'agent',
            'message': response_data['response'],
            'metadata': response_data['metadata']
        }
        st.session_state.conversation_history.append(agent_message)
    
    # Store analytics data
    analytics_entry = create_response_summary({
        'response': response_data['response'] if response_data else '',
        'metadata': response_data['metadata'] if response_data else {}
    })
    analytics_entry['processing_time'] = processing_time
    st.session_state.analytics_data.append(analytics_entry)

def display_analysis_results(customer_message, sentiment_analysis, escalation_prediction, 
                           source_docs, response_data, processing_time, show_debug):
    """Display comprehensive analysis results"""
    
    st.subheader("📊 Analysis Results")
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment = sentiment_analysis['overall_sentiment']
        sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        st.metric("Overall Sentiment", sentiment, delta=None)
        st.write(f"{sentiment_emoji.get(sentiment, '😐')} {sentiment.title()}")
    
    with col2:
        urgency = sentiment_analysis['urgency']
        urgency_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        st.metric("Urgency Level", urgency)
        st.write(f"{urgency_colors.get(urgency, '🟢')} {urgency.title()}")
    
    with col3:
        escalation_risk = escalation_prediction['risk_level']
        risk_colors = {"high": "🚨", "medium": "⚠️", "low": "✅"}
        st.metric("Escalation Risk", escalation_risk)
        st.write(f"{risk_colors.get(escalation_risk, '✅')} {escalation_risk.title()}")
    
    with col4:
        st.metric("Processing Time", f"{processing_time:.2f}s")
        st.write("⚡ Real-time analysis")
    
    # Detailed analysis tabs
    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
        "🎭 Sentiment Details", "⚠️ Escalation Analysis", "📚 Knowledge Sources", "💬 Generated Response"
    ])
    
    with analysis_tab1:
        display_sentiment_details(sentiment_analysis)
    
    with analysis_tab2:
        display_escalation_details(escalation_prediction)
    
    with analysis_tab3:
        display_knowledge_sources(source_docs)
    
    with analysis_tab4:
        if response_data:
            display_generated_response(response_data)
        else:
            st.info("Auto-response generation is disabled. Enable it in the sidebar to see generated responses.")
    
    # Debug information
    if show_debug:
        with st.expander("🔍 Debug Information"):
            st.json({
                'sentiment_analysis': sentiment_analysis,
                'escalation_prediction': escalation_prediction,
                'processing_time': processing_time
            })

def display_sentiment_details(sentiment_analysis):
    """Display detailed sentiment analysis"""
    
    emotions = sentiment_analysis['emotions']
    
    # Emotion chart
    if emotions:
        emotion_df = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Score'])
        emotion_df = emotion_df[emotion_df['Score'] > 0.1]  # Only show significant emotions
        
        if not emotion_df.empty:
            fig = px.bar(emotion_df, x='Emotion', y='Score', 
                        title="Detected Emotions", 
                        color='Score',
                        color_continuous_scale='RdYlBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment scores
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("VADER Scores")
        vader_scores = sentiment_analysis['scores']['vader']
        st.metric("Compound", f"{vader_scores['compound']:.3f}")
        st.metric("Positive", f"{vader_scores['positive']:.3f}")
        st.metric("Negative", f"{vader_scores['negative']:.3f}")
        st.metric("Neutral", f"{vader_scores['neutral']:.3f}")
    
    with col2:
        st.subheader("TextBlob Scores")
        textblob_scores = sentiment_analysis['scores']['textblob']
        st.metric("Polarity", f"{textblob_scores['polarity']:.3f}")
        st.metric("Subjectivity", f"{textblob_scores['subjectivity']:.3f}")
        
        # Confidence and text stats
        st.metric("Confidence", f"{sentiment_analysis['confidence']:.3f}")
        st.metric("Word Count", sentiment_analysis['word_count'])

def display_escalation_details(escalation_prediction):
    """Display detailed escalation analysis"""
    
    risk_level = escalation_prediction['risk_level']
    escalation_class = f"escalation-{risk_level.split('-')[0] if '-' in risk_level else risk_level}"
    
    st.markdown(f"""
    <div class="sentiment-card {escalation_class}">
        <h4>Escalation Risk: {risk_level.title()}</h4>
        <p><strong>Probability:</strong> {escalation_prediction['escalation_probability']:.1%}</p>
        <p><strong>Confidence:</strong> {escalation_prediction['confidence']:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key indicators
    if escalation_prediction['key_indicators']:
        st.subheader("🚩 Key Risk Indicators")
        for indicator in escalation_prediction['key_indicators']:
            st.write(f"• {indicator.replace('_', ' ').title()}")
    
    # Recommendations
    if escalation_prediction['recommended_actions']:
        st.subheader("💡 Recommended Actions")
        for action in escalation_prediction['recommended_actions']:
            st.write(action)
    
    # Pattern analysis
    patterns = escalation_prediction.get('patterns_detected', {})
    if patterns:
        st.subheader("📈 Pattern Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Language Score", f"{patterns.get('language_score', 0):.3f}")
            st.metric("Behavioral Score", f"{patterns.get('behavioral_score', 0):.3f}")
        
        with col2:
            st.metric("Sentiment Score", f"{patterns.get('sentiment_score', 0):.3f}")
            st.metric("Historical Modifier", f"{patterns.get('historical_modifier', 1):.3f}")

def display_knowledge_sources(source_docs):
    """Display knowledge base sources"""
    
    if not source_docs:
        st.info("No relevant knowledge base articles found.")
        return
    
    st.subheader(f"📚 Retrieved {len(source_docs)} Relevant Articles")
    
    for i, doc in enumerate(source_docs, 1):
        with st.expander(f"📄 {i}. {doc['title']} (Score: {doc['score']:.3f})"):
            st.write(f"**Category:** {doc['category']}")
            st.write("**Preview:**")
            st.write(doc['content_preview'])

def display_generated_response(response_data):
    """Display the generated response and metadata"""
    
    response = response_data['response']
    metadata = response_data['metadata']
    
    # Response text
    st.subheader("💬 Generated Response")
    st.write(response)
    
    # Response metadata
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Response Metadata")
        st.write(f"**Emotion Detected:** {metadata['emotion_detected']}")
        st.write(f"**Urgency Level:** {metadata['urgency_level']}")
        st.write(f"**Escalation Risk:** {metadata['escalation_risk']}")
        st.write(f"**Response Strategy:** {metadata['response_strategy']}")
    
    with col2:
        st.subheader("🎛️ Tone Adjustments")
        tone_adjustments = metadata.get('tone_adjustments', [])
        if tone_adjustments:
            for adjustment in tone_adjustments:
                st.write(f"• {adjustment.replace('_', ' ').title()}")
        else:
            st.write("No special tone adjustments applied")
        
        st.write(f"**Knowledge Sources Used:** {metadata['knowledge_sources_used']}")

def analytics_dashboard(system):
    """Analytics and insights dashboard"""
    
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.analytics_data:
        st.info("No analytics data available yet. Process some customer messages to see insights!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.analytics_data)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Interactions", len(df))
    
    with col2:
        avg_processing_time = df['processing_time'].mean() if 'processing_time' in df else 0
        st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
    
    with col3:
        high_risk_count = len(df[df['escalation_risk'] == 'high']) if 'escalation_risk' in df else 0
        st.metric("High Risk Interactions", high_risk_count)
    
    with col4:
        avg_response_length = df['response_length'].mean() if 'response_length' in df else 0
        st.metric("Avg Response Length", f"{avg_response_length:.0f} chars")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        if 'emotion_detected' in df:
            sentiment_counts = df['emotion_detected'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                        title="Emotion Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Escalation risk distribution
        if 'escalation_risk' in df:
            risk_counts = df['escalation_risk'].value_counts()
            fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                        title="Escalation Risk Distribution",
                        color=risk_counts.values,
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed data table
    st.subheader("📋 Interaction Details")
    st.dataframe(df, use_container_width=True)

def knowledge_base_interface(rag_system):
    """Knowledge base management interface"""
    
    st.header("📚 Knowledge Base Management")
    
    # Knowledge base stats
    stats = rag_system.get_collection_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Articles", stats.get('total_documents', 0))
    with col2:
        st.metric("Categories", len(stats.get('categories', {})))
    with col3:
        st.metric("Embedding Model", stats.get('embedding_model', 'Unknown'))
    
    # Category breakdown
    if stats.get('categories'):
        st.subheader("📂 Category Breakdown")
        categories_df = pd.DataFrame(list(stats['categories'].items()), 
                                   columns=['Category', 'Count'])
        fig = px.bar(categories_df, x='Category', y='Count', 
                    title="Articles by Category")
        st.plotly_chart(fig, use_container_width=True)
    
    # Search interface
    st.subheader("🔍 Search Knowledge Base")
    search_query = st.text_input("Search Query", placeholder="Enter your search terms...")
    
    if search_query:
        with st.spinner("Searching..."):
            results = rag_system.search(search_query, n_results=5)
        
        if results:
            st.subheader(f"Found {len(results)} Results")
            for i, result in enumerate(results, 1):
                with st.expander(f"{i}. {result['metadata'].get('title', 'Unknown')} (Score: {result['score']:.3f})"):
                    st.write(f"**Category:** {result['metadata'].get('category', 'Unknown')}")
                    st.write("**Content:**")
                    st.write(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])
        else:
            st.info("No results found for your search query.")

def testing_interface(system):
    """Testing and experimentation interface"""
    
    st.header("🧪 Testing Interface")
    
    # Predefined test messages
    test_messages = {
        "Angry Customer": "I AM SO ANGRY! Your service is terrible and I want to cancel my subscription RIGHT NOW! This is the worst customer service ever!",
        "Confused Customer": "Hi, I'm not sure how to use the new feature. Could you help me understand how it works?",
        "Happy Customer": "I love your service! It's amazing and has helped my business so much. Thank you!",
        "Security Concern": "I'm worried that my account might have been hacked. I saw some suspicious activity.",
        "Legal Threat": "I'm going to sue you and report this to the Better Business Bureau if you don't fix this immediately!",
        "Technical Issue": "The app keeps crashing when I try to upload files. It's very frustrating."
    }
    
    st.subheader("📝 Quick Test Messages")
    
    selected_test = st.selectbox("Choose a test scenario:", list(test_messages.keys()))
    
    if st.button("Load Test Message"):
        st.session_state['test_message'] = test_messages[selected_test]
        st.rerun()
    
    # Test message input
    test_message = st.text_area(
        "Test Message:",
        value=st.session_state.get('test_message', ''),
        height=100,
        key="test_message_input"
    )
    
    if st.button("🧪 Run Test Analysis") and test_message:
        with st.spinner("Running analysis..."):
            # Run all analyses
            sentiment = system['sentiment_analyzer'].analyze_sentiment(test_message)
            escalation = system['escalation_predictor'].predict_escalation(
                customer_id="test_customer",
                message=test_message,
                sentiment_analysis=sentiment
            )
            context, sources = system['rag_system'].get_context(test_message)
            response = system['response_generator'].generate_response(
                customer_message=test_message,
                sentiment_analysis=sentiment,
                escalation_prediction=escalation,
                knowledge_context=context,
                source_documents=sources
            )
        
        # Display results in compact format
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Analysis Results")
            st.write(f"**Sentiment:** {sentiment['overall_sentiment']}")
            st.write(f"**Dominant Emotion:** {max(sentiment['emotions'], key=sentiment['emotions'].get)}")
            st.write(f"**Urgency:** {sentiment['urgency']}")
            st.write(f"**Escalation Risk:** {escalation['risk_level']}")
            st.write(f"**Confidence:** {escalation['confidence']:.1%}")
        
        with col2:
            st.subheader("💬 Generated Response")
            st.write(response['response'])
        
        # Show key indicators
        if escalation['key_indicators']:
            st.subheader("🚩 Key Indicators")
            for indicator in escalation['key_indicators']:
                st.write(f"• {indicator}")

if __name__ == "__main__":
    main() 