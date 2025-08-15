"""
Escalation Predictor Module for Customer Support
Advanced pattern recognition for predicting customer escalation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EscalationPredictor:
    """
    Advanced escalation pattern recognition and prediction system
    """
    
    def __init__(self):
        """Initialize the escalation predictor"""
        
        # Escalation indicators with weights
        self.escalation_indicators = {
            # Language patterns
            'threat_keywords': {
                'patterns': [
                    'lawsuit', 'lawyer', 'attorney', 'sue', 'legal action', 'court',
                    'bbb', 'better business bureau', 'report you', 'file complaint'
                ],
                'weight': 0.3
            },
            'cancellation_threats': {
                'patterns': [
                    'cancel', 'canceling', 'unsubscribe', 'switch provider',
                    'find alternative', 'done with', 'had enough'
                ],
                'weight': 0.25
            },
            'authority_demands': {
                'patterns': [
                    'manager', 'supervisor', 'speak to', 'escalate',
                    'higher up', 'someone in charge', 'your boss'
                ],
                'weight': 0.2
            },
            'public_exposure_threats': {
                'patterns': [
                    'social media', 'twitter', 'facebook', 'instagram',
                    'review', 'yelp', 'google reviews', 'tell everyone',
                    'post online', 'viral', 'expose'
                ],
                'weight': 0.25
            },
            'extreme_language': {
                'patterns': [
                    'worst', 'terrible', 'awful', 'horrible', 'disgusting',
                    'pathetic', 'useless', 'incompetent', 'idiots', 'morons'
                ],
                'weight': 0.15
            },
            'urgency_demands': {
                'patterns': [
                    'immediately', 'right now', 'asap', 'urgent',
                    'emergency', 'critical', 'can\'t wait'
                ],
                'weight': 0.1
            }
        }
        
        # Behavioral patterns
        self.behavioral_patterns = {
            'repeat_contact': {'threshold': 3, 'weight': 0.2},
            'rapid_succession': {'threshold': 2, 'timeframe_hours': 1, 'weight': 0.15},
            'channel_switching': {'threshold': 2, 'weight': 0.1},
            'long_messages': {'threshold': 500, 'weight': 0.05},
            'caps_usage': {'threshold': 0.3, 'weight': 0.1}
        }
        
        # Historical data for pattern learning
        self.interaction_history = defaultdict(list)
        self.escalation_history = []
        
        logger.info("Escalation predictor initialized")
    
    def predict_escalation(
        self,
        customer_id: str,
        message: str,
        conversation_history: List[Dict[str, Any]] = None,
        sentiment_analysis: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Predict escalation likelihood for a customer interaction
        
        Args:
            customer_id: Unique customer identifier
            message: Current customer message
            conversation_history: Previous messages in conversation
            sentiment_analysis: Results from sentiment analysis
            
        Returns:
            Dictionary containing escalation prediction and analysis
        """
        
        # Initialize prediction result
        prediction = {
            'escalation_probability': 0.0,
            'risk_level': 'low',
            'confidence': 0.0,
            'key_indicators': [],
            'recommended_actions': [],
            'patterns_detected': {}
        }
        
        # Analyze language patterns
        language_score, language_indicators = self._analyze_language_patterns(message)
        
        # Analyze behavioral patterns
        behavioral_score, behavioral_patterns = self._analyze_behavioral_patterns(
            customer_id, message, conversation_history
        )
        
        # Incorporate sentiment analysis if available
        sentiment_score = self._incorporate_sentiment_analysis(sentiment_analysis)
        
        # Calculate overall escalation probability
        base_probability = (
            language_score * 0.5 +
            behavioral_score * 0.3 +
            sentiment_score * 0.2
        )
        
        # Apply historical patterns if available
        historical_modifier = self._get_historical_modifier(customer_id)
        
        # Final probability calculation
        final_probability = min(base_probability * historical_modifier, 1.0)
        
        # Determine risk level and confidence
        risk_level, confidence = self._classify_risk_level(final_probability)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            language_indicators,
            behavioral_patterns,
            sentiment_analysis,
            risk_level
        )
        
        # Update prediction result
        prediction.update({
            'escalation_probability': round(final_probability, 3),
            'risk_level': risk_level,
            'confidence': confidence,
            'key_indicators': language_indicators + list(behavioral_patterns.keys()),
            'recommended_actions': recommendations,
            'patterns_detected': {
                'language_score': language_score,
                'behavioral_score': behavioral_score,
                'sentiment_score': sentiment_score,
                'historical_modifier': historical_modifier
            }
        })
        
        # Store interaction for learning
        self._store_interaction(customer_id, message, prediction)
        
        return prediction
    
    def _analyze_language_patterns(self, message: str) -> Tuple[float, List[str]]:
        """Analyze language patterns in the message"""
        message_lower = message.lower()
        total_score = 0.0
        detected_indicators = []
        
        for category, config in self.escalation_indicators.items():
            patterns = config['patterns']
            weight = config['weight']
            
            # Count pattern matches
            matches = sum(1 for pattern in patterns if pattern in message_lower)
            
            if matches > 0:
                # Calculate category score
                category_score = min(matches * weight, weight)
                total_score += category_score
                detected_indicators.append(category)
        
        # Additional analysis
        
        # Check for excessive punctuation
        exclamation_count = message.count('!')
        question_count = message.count('?')
        if exclamation_count > 3 or question_count > 5:
            total_score += 0.1
            detected_indicators.append('excessive_punctuation')
        
        # Check for repetitive text
        words = message_lower.split()
        if len(words) > 0:
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / len(words))
            if repetition_ratio > 0.3:
                total_score += 0.05
                detected_indicators.append('repetitive_language')
        
        return min(total_score, 1.0), detected_indicators
    
    def _analyze_behavioral_patterns(
        self,
        customer_id: str,
        message: str,
        conversation_history: List[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Analyze behavioral patterns"""
        
        behavioral_score = 0.0
        detected_patterns = {}
        
        if not conversation_history:
            conversation_history = []
        
        # Add current message to history for analysis
        current_interaction = {
            'message': message,
            'timestamp': datetime.now(),
            'customer_id': customer_id
        }
        
        # Analyze message length
        if len(message) > self.behavioral_patterns['long_messages']['threshold']:
            behavioral_score += self.behavioral_patterns['long_messages']['weight']
            detected_patterns['long_message'] = True
        
        # Analyze caps usage
        if message:
            caps_ratio = sum(1 for c in message if c.isupper()) / len(message)
            if caps_ratio > self.behavioral_patterns['caps_usage']['threshold']:
                behavioral_score += self.behavioral_patterns['caps_usage']['weight']
                detected_patterns['excessive_caps'] = True
        
        # Analyze conversation patterns if history is available
        if conversation_history:
            
            # Check for repeat contact
            message_count = len(conversation_history) + 1  # +1 for current message
            if message_count >= self.behavioral_patterns['repeat_contact']['threshold']:
                behavioral_score += self.behavioral_patterns['repeat_contact']['weight']
                detected_patterns['repeat_contact'] = message_count
            
            # Check for rapid succession
            recent_messages = [msg for msg in conversation_history 
                             if 'timestamp' in msg and 
                             datetime.now() - msg['timestamp'] <= timedelta(hours=1)]
            
            if len(recent_messages) >= self.behavioral_patterns['rapid_succession']['threshold']:
                behavioral_score += self.behavioral_patterns['rapid_succession']['weight']
                detected_patterns['rapid_succession'] = len(recent_messages)
        
        return min(behavioral_score, 1.0), detected_patterns
    
    def _incorporate_sentiment_analysis(self, sentiment_analysis: Dict[str, Any] = None) -> float:
        """Incorporate sentiment analysis into escalation prediction"""
        if not sentiment_analysis:
            return 0.0
        
        sentiment_score = 0.0
        
        # Negative sentiment increases escalation risk
        overall_sentiment = sentiment_analysis.get('overall_sentiment', 'neutral')
        if overall_sentiment == 'negative':
            sentiment_score += 0.3
        
        # High negative emotions increase risk
        emotions = sentiment_analysis.get('emotions', {})
        if emotions.get('anger', 0) > 0.6:
            sentiment_score += 0.4
        if emotions.get('fear', 0) > 0.5:
            sentiment_score += 0.2
        if emotions.get('disgust', 0) > 0.5:
            sentiment_score += 0.3
        
        # Existing escalation risk assessment
        escalation_risk = sentiment_analysis.get('escalation_risk', {})
        if escalation_risk.get('level') == 'high':
            sentiment_score += 0.4
        elif escalation_risk.get('level') == 'medium':
            sentiment_score += 0.2
        
        # Urgency level
        urgency = sentiment_analysis.get('urgency', 'low')
        if urgency == 'high':
            sentiment_score += 0.2
        elif urgency == 'medium':
            sentiment_score += 0.1
        
        return min(sentiment_score, 1.0)
    
    def _get_historical_modifier(self, customer_id: str) -> float:
        """Get historical modifier based on customer's past escalations"""
        # If no history, return neutral modifier
        if customer_id not in self.interaction_history:
            return 1.0
        
        customer_interactions = self.interaction_history[customer_id]
        
        if len(customer_interactions) < 3:
            return 1.0
        
        # Count recent escalations
        recent_escalations = sum(
            1 for interaction in customer_interactions[-10:]  # Last 10 interactions
            if interaction.get('escalated', False)
        )
        
        # Calculate modifier
        if recent_escalations >= 3:
            return 1.3  # Increase probability
        elif recent_escalations >= 1:
            return 1.1
        else:
            return 0.9  # Slightly decrease if good history
    
    def _classify_risk_level(self, probability: float) -> Tuple[str, float]:
        """Classify risk level and calculate confidence"""
        
        if probability >= 0.7:
            risk_level = "high"
            confidence = 0.9
        elif probability >= 0.4:
            risk_level = "medium"
            confidence = 0.7
        elif probability >= 0.2:
            risk_level = "low-medium"
            confidence = 0.6
        else:
            risk_level = "low"
            confidence = 0.8
        
        return risk_level, confidence
    
    def _generate_recommendations(
        self,
        language_indicators: List[str],
        behavioral_patterns: Dict[str, Any],
        sentiment_analysis: Dict[str, Any] = None,
        risk_level: str = 'low'
    ) -> List[str]:
        """Generate action recommendations based on analysis"""
        
        recommendations = []
        
        # High-risk recommendations
        if risk_level == "high":
            recommendations.extend([
                "🚨 Immediate supervisor notification required",
                "📞 Consider phone call instead of text response",
                "🎯 Assign to senior support agent",
                "📝 Document all interactions carefully"
            ])
        
        # Threat-related recommendations
        if 'threat_keywords' in language_indicators:
            recommendations.extend([
                "⚖️ Legal team notification may be required",
                "📋 Escalate to management immediately",
                "🔒 Follow escalation protocols"
            ])
        
        # Cancellation threat recommendations
        if 'cancellation_threats' in language_indicators:
            recommendations.extend([
                "💰 Consider retention offers",
                "🤝 Schedule call with account manager",
                "📊 Review account value and history"
            ])
        
        # Authority demand recommendations
        if 'authority_demands' in language_indicators:
            recommendations.extend([
                "👥 Connect with supervisor/manager",
                "📞 Expedite response time",
                "🎖️ Assign senior team member"
            ])
        
        # Public exposure threat recommendations
        if 'public_exposure_threats' in language_indicators:
            recommendations.extend([
                "📱 Social media team notification",
                "🏃 Rapid response required",
                "💬 Consider public response strategy"
            ])
        
        # Behavioral pattern recommendations
        if 'repeat_contact' in behavioral_patterns:
            recommendations.append("🔄 Review previous interactions for context")
        
        if 'rapid_succession' in behavioral_patterns:
            recommendations.append("⚡ Prioritize response - customer showing urgency")
        
        if 'excessive_caps' in behavioral_patterns:
            recommendations.append("🗣️ Acknowledge customer's frustration explicitly")
        
        # Sentiment-based recommendations
        if sentiment_analysis:
            emotions = sentiment_analysis.get('emotions', {})
            
            if emotions.get('anger', 0) > 0.6:
                recommendations.append("😤 Use de-escalation techniques")
            
            if emotions.get('fear', 0) > 0.5:
                recommendations.append("🤗 Provide reassurance and clear next steps")
            
            if sentiment_analysis.get('urgency') == 'high':
                recommendations.append("⏰ Expedited handling required")
        
        # Default recommendations if none specific
        if not recommendations:
            recommendations.extend([
                "💬 Provide empathetic response",
                "🔍 Gather additional context if needed",
                "✅ Follow standard support procedures"
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _store_interaction(self, customer_id: str, message: str, prediction: Dict[str, Any]):
        """Store interaction for historical analysis"""
        interaction = {
            'timestamp': datetime.now(),
            'message': message,
            'prediction': prediction,
            'escalated': False  # This would be updated later if escalation occurs
        }
        
        self.interaction_history[customer_id].append(interaction)
        
        # Keep only last 50 interactions per customer
        if len(self.interaction_history[customer_id]) > 50:
            self.interaction_history[customer_id] = self.interaction_history[customer_id][-50:]
    
    def record_escalation(self, customer_id: str, escalated: bool = True):
        """Record whether the latest interaction led to escalation"""
        if customer_id in self.interaction_history and self.interaction_history[customer_id]:
            self.interaction_history[customer_id][-1]['escalated'] = escalated
            
            # Add to escalation history for model improvement
            self.escalation_history.append({
                'customer_id': customer_id,
                'timestamp': datetime.now(),
                'escalated': escalated
            })
    
    def get_escalation_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics and insights"""
        total_interactions = sum(len(history) for history in self.interaction_history.values())
        total_escalations = sum(1 for history in self.interaction_history.values() 
                              for interaction in history if interaction.get('escalated', False))
        
        escalation_rate = total_escalations / total_interactions if total_interactions > 0 else 0
        
        # Most common escalation indicators
        indicator_counts = defaultdict(int)
        for history in self.interaction_history.values():
            for interaction in history:
                if interaction.get('escalated', False):
                    indicators = interaction.get('prediction', {}).get('key_indicators', [])
                    for indicator in indicators:
                        indicator_counts[indicator] += 1
        
        top_indicators = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_interactions': total_interactions,
            'total_escalations': total_escalations,
            'escalation_rate': round(escalation_rate, 3),
            'unique_customers': len(self.interaction_history),
            'top_escalation_indicators': top_indicators
        } 