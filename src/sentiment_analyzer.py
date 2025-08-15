"""
Sentiment Analysis Module for Customer Support
Multi-model approach for emotion detection and mood analysis
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Multi-model sentiment analysis system for customer support
    """
    
    def __init__(self, use_transformer: bool = True):
        """
        Initialize sentiment analyzer with multiple models
        
        Args:
            use_transformer: Whether to load transformer-based emotion model
        """
        self.use_transformer = use_transformer
        
        # Initialize VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize transformer-based emotion classifier if requested
        if use_transformer:
            self._init_emotion_classifier()
        
        # Define emotion keywords for rule-based detection
        self.emotion_keywords = {
            'anger': ['angry', 'mad', 'furious', 'outraged', 'annoyed', 'irritated', 'frustrated', 'pissed'],
            'joy': ['happy', 'excited', 'pleased', 'satisfied', 'delighted', 'thrilled', 'grateful'],
            'sadness': ['sad', 'disappointed', 'upset', 'depressed', 'miserable', 'heartbroken'],
            'fear': ['worried', 'anxious', 'scared', 'nervous', 'concerned', 'afraid', 'terrified'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'disgust': ['disgusted', 'revolted', 'appalled', 'horrified', 'sickened']
        }
        
        # Define escalation indicators
        self.escalation_keywords = [
            'lawsuit', 'lawyer', 'attorney', 'sue', 'legal action', 'court',
            'cancel', 'canceling', 'unsubscribe', 'refund', 'money back',
            'terrible', 'awful', 'worst', 'horrible', 'disgusting', 'pathetic',
            'manager', 'supervisor', 'complaint', 'report', 'better business bureau',
            'social media', 'twitter', 'facebook', 'review', 'yelp'
        ]
        
        logger.info("Sentiment analyzer initialized successfully")
    
    def _init_emotion_classifier(self):
        """Initialize transformer-based emotion classifier"""
        try:
            # Use a lightweight emotion classification model
            model_name = "j-hartmann/emotion-english-distilroberta-base"
            
            logger.info(f"Loading emotion classifier: {model_name}")
            self.emotion_classifier = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True,
                truncation=True,
                max_length=512
            )
            logger.info("Emotion classifier loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load emotion classifier: {e}")
            logger.info("Falling back to rule-based emotion detection")
            self.emotion_classifier = None
            self.use_transformer = False
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis using multiple methods
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment scores and analysis
        """
        if not text or not text.strip():
            return self._empty_sentiment_result()
        
        # Clean text
        cleaned_text = self._preprocess_text(text)
        
        # VADER sentiment analysis
        vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(cleaned_text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Emotion analysis
        emotions = self._analyze_emotions(cleaned_text)
        
        # Escalation risk assessment
        escalation_risk = self._assess_escalation_risk(cleaned_text)
        
        # Overall sentiment classification
        overall_sentiment = self._classify_overall_sentiment(
            vader_scores['compound'],
            textblob_polarity
        )
        
        # Confidence score
        confidence = self._calculate_confidence(vader_scores, textblob_polarity)
        
        # Urgency level
        urgency = self._assess_urgency(cleaned_text, emotions, escalation_risk)
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': confidence,
            'urgency': urgency,
            'emotions': emotions,
            'escalation_risk': escalation_risk,
            'scores': {
                'vader': {
                    'compound': vader_scores['compound'],
                    'positive': vader_scores['pos'],
                    'negative': vader_scores['neg'],
                    'neutral': vader_scores['neu']
                },
                'textblob': {
                    'polarity': textblob_polarity,
                    'subjectivity': textblob_subjectivity
                }
            },
            'text_length': len(text),
            'word_count': len(text.split())
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        
        return text.strip()
    
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """
        Analyze emotions using transformer model and/or rule-based approach
        """
        emotions = {
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0
        }
        
        # Transformer-based emotion detection
        if self.use_transformer and self.emotion_classifier:
            try:
                results = self.emotion_classifier(text)
                if results and len(results) > 0:
                    for result in results[0]:
                        emotion_label = result['label'].lower()
                        score = result['score']
                        
                        # Map model labels to our emotion categories
                        emotion_mapping = {
                            'joy': 'joy',
                            'sadness': 'sadness',
                            'anger': 'anger',
                            'fear': 'fear',
                            'surprise': 'surprise',
                            'disgust': 'disgust'
                        }
                        
                        if emotion_label in emotion_mapping:
                            emotions[emotion_mapping[emotion_label]] = score
                            
            except Exception as e:
                logger.warning(f"Transformer emotion analysis failed: {e}")
        
        # Rule-based emotion detection as backup or supplement
        rule_emotions = self._rule_based_emotion_detection(text)
        
        # Combine transformer and rule-based results
        for emotion in emotions:
            if emotion in rule_emotions:
                # Take max of transformer and rule-based scores
                emotions[emotion] = max(emotions[emotion], rule_emotions[emotion])
        
        return emotions
    
    def _rule_based_emotion_detection(self, text: str) -> Dict[str, float]:
        """Rule-based emotion detection using keyword matching"""
        emotions = {emotion: 0.0 for emotion in self.emotion_keywords}
        
        words = text.split()
        total_words = len(words)
        
        if total_words == 0:
            return emotions
        
        for emotion, keywords in self.emotion_keywords.items():
            matches = sum(1 for word in words if any(kw in word for kw in keywords))
            emotions[emotion] = min(matches / total_words * 10, 1.0)  # Normalize to 0-1
        
        return emotions
    
    def _assess_escalation_risk(self, text: str) -> Dict[str, Any]:
        """Assess risk of customer escalation"""
        escalation_score = 0.0
        triggers = []
        
        # Check for escalation keywords
        for keyword in self.escalation_keywords:
            if keyword in text:
                escalation_score += 0.2
                triggers.append(keyword)
        
        # Check for all caps (shouting)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            escalation_score += 0.3
            triggers.append("excessive_caps")
        
        # Check for multiple exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count > 2:
            escalation_score += 0.2
            triggers.append("multiple_exclamations")
        
        # Check for negative sentiment intensity
        vader_score = self.vader_analyzer.polarity_scores(text)['compound']
        if vader_score < -0.5:
            escalation_score += 0.3
            triggers.append("very_negative_sentiment")
        
        # Normalize escalation score
        escalation_score = min(escalation_score, 1.0)
        
        # Classify risk level
        if escalation_score >= 0.7:
            risk_level = "high"
        elif escalation_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            'score': escalation_score,
            'level': risk_level,
            'triggers': triggers
        }
    
    def _classify_overall_sentiment(self, vader_compound: float, textblob_polarity: float) -> str:
        """Classify overall sentiment based on multiple scores"""
        # Average the scores
        avg_score = (vader_compound + textblob_polarity) / 2
        
        if avg_score >= 0.05:
            return "positive"
        elif avg_score <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_confidence(self, vader_scores: Dict, textblob_polarity: float) -> float:
        """Calculate confidence in sentiment analysis"""
        # Higher confidence when models agree
        vader_sentiment = "positive" if vader_scores['compound'] > 0.05 else "negative" if vader_scores['compound'] < -0.05 else "neutral"
        textblob_sentiment = "positive" if textblob_polarity > 0.05 else "negative" if textblob_polarity < -0.05 else "neutral"
        
        agreement_bonus = 0.3 if vader_sentiment == textblob_sentiment else 0.0
        
        # Base confidence on absolute values
        vader_confidence = abs(vader_scores['compound'])
        textblob_confidence = abs(textblob_polarity)
        
        base_confidence = (vader_confidence + textblob_confidence) / 2
        final_confidence = min(base_confidence + agreement_bonus, 1.0)
        
        return round(final_confidence, 3)
    
    def _assess_urgency(self, text: str, emotions: Dict[str, float], escalation_risk: Dict) -> str:
        """Assess urgency level of customer message"""
        urgency_score = 0.0
        
        # High negative emotions increase urgency
        if emotions['anger'] > 0.6 or emotions['fear'] > 0.6:
            urgency_score += 0.4
        
        # Escalation risk increases urgency
        urgency_score += escalation_risk['score'] * 0.4
        
        # Urgent keywords
        urgent_keywords = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'help']
        if any(keyword in text for keyword in urgent_keywords):
            urgency_score += 0.3
        
        # Classify urgency
        if urgency_score >= 0.7:
            return "high"
        elif urgency_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _empty_sentiment_result(self) -> Dict[str, Any]:
        """Return empty sentiment analysis result"""
        return {
            'overall_sentiment': 'neutral',
            'confidence': 0.0,
            'urgency': 'low',
            'emotions': {emotion: 0.0 for emotion in self.emotion_keywords},
            'escalation_risk': {
                'score': 0.0,
                'level': 'low',
                'triggers': []
            },
            'scores': {
                'vader': {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
                'textblob': {'polarity': 0.0, 'subjectivity': 0.0}
            },
            'text_length': 0,
            'word_count': 0
        }
    
    def analyze_conversation_trend(self, messages: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment trend across multiple messages in a conversation
        
        Args:
            messages: List of messages in chronological order
            
        Returns:
            Dictionary containing trend analysis
        """
        if not messages:
            return {'trend': 'stable', 'sentiment_history': [], 'escalation_trend': 'stable'}
        
        sentiment_history = []
        escalation_scores = []
        
        for message in messages:
            analysis = self.analyze_sentiment(message)
            sentiment_history.append({
                'sentiment': analysis['overall_sentiment'],
                'score': analysis['scores']['vader']['compound'],
                'escalation_risk': analysis['escalation_risk']['score']
            })
            escalation_scores.append(analysis['escalation_risk']['score'])
        
        # Calculate trends
        sentiment_trend = self._calculate_sentiment_trend(sentiment_history)
        escalation_trend = self._calculate_escalation_trend(escalation_scores)
        
        return {
            'trend': sentiment_trend,
            'escalation_trend': escalation_trend,
            'sentiment_history': sentiment_history,
            'message_count': len(messages),
            'latest_sentiment': sentiment_history[-1] if sentiment_history else None
        }
    
    def _calculate_sentiment_trend(self, sentiment_history: List[Dict]) -> str:
        """Calculate overall sentiment trend"""
        if len(sentiment_history) < 2:
            return 'stable'
        
        scores = [s['score'] for s in sentiment_history]
        
        # Simple trend calculation
        recent_avg = np.mean(scores[-3:]) if len(scores) >= 3 else scores[-1]
        early_avg = np.mean(scores[:3]) if len(scores) >= 3 else scores[0]
        
        diff = recent_avg - early_avg
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_escalation_trend(self, escalation_scores: List[float]) -> str:
        """Calculate escalation risk trend"""
        if len(escalation_scores) < 2:
            return 'stable'
        
        recent_avg = np.mean(escalation_scores[-3:]) if len(escalation_scores) >= 3 else escalation_scores[-1]
        early_avg = np.mean(escalation_scores[:3]) if len(escalation_scores) >= 3 else escalation_scores[0]
        
        diff = recent_avg - early_avg
        
        if diff > 0.1:
            return 'escalating'
        elif diff < -0.1:
            return 'de-escalating'
        else:
            return 'stable' 