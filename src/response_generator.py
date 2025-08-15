"""
Response Generator Module for Customer Support
Empathetic response generation with tone calibration
"""

import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple
import random
from datetime import datetime
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Empathetic response generator with tone calibration
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the response generator
        
        Args:
            openai_api_key: OpenAI API key (optional)
        """
        # Set up OpenAI if API key is available
        self.use_openai = False
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            try:
                openai.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
                self.use_openai = True
                logger.info("OpenAI API initialized for enhanced response generation")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI API: {e}")
        
        # Response templates by emotion and urgency
        self.response_templates = {
            'anger': {
                'high': [
                    "I completely understand your frustration, and I sincerely apologize for the inconvenience you've experienced. Let me personally ensure this gets resolved immediately.",
                    "I can see how upset you are, and you have every right to feel this way. This is absolutely not the experience we want for our valued customers. Let me make this right for you right now.",
                    "Your frustration is completely justified, and I'm truly sorry this happened. I'm going to prioritize your case and work directly with my supervisor to resolve this today."
                ],
                'medium': [
                    "I understand this situation is frustrating, and I apologize for any inconvenience. Let me help you resolve this quickly.",
                    "I can see why this would be annoying, and I'm sorry you're dealing with this. Let's get this sorted out for you.",
                    "I completely understand your frustration. This isn't the experience we want for you, and I'm here to help fix it."
                ],
                'low': [
                    "I understand your concern and I'm here to help resolve this for you.",
                    "Thank you for bringing this to our attention. Let me see what I can do to help.",
                    "I apologize for any inconvenience. Let's work together to solve this."
                ]
            },
            'sadness': {
                'high': [
                    "I'm so sorry you're going through this. I can hear how much this means to you, and I want you to know that I'm here to support you every step of the way.",
                    "This must be really difficult for you, and I'm genuinely sorry you're experiencing this. Let me do everything I can to help make this better.",
                    "I can sense how disappointed you are, and I truly empathize with your situation. You're not alone in this - I'm going to work hard to find a solution."
                ],
                'medium': [
                    "I'm sorry to hear you're feeling disappointed about this. Let me see how I can help improve the situation.",
                    "I understand this isn't what you were hoping for, and I'm sorry about that. Let's work together to find a better solution.",
                    "I can see this is upsetting, and I'm sorry you're dealing with this. I'm here to help however I can."
                ],
                'low': [
                    "I understand your disappointment and I'm here to help.",
                    "I'm sorry this didn't meet your expectations. Let me see what options we have.",
                    "I hear your concern and I'd like to help resolve this for you."
                ]
            },
            'fear': {
                'high': [
                    "I want to reassure you that you're in good hands. I understand your concerns, and I'm going to personally ensure everything is taken care of properly.",
                    "Please don't worry - we're going to resolve this together. I'll stay with you through this process and keep you informed every step of the way.",
                    "I can understand why this would be concerning, and I want to put your mind at ease. Let me explain exactly what we're going to do to fix this."
                ],
                'medium': [
                    "I understand your concerns, and I want to reassure you that we can resolve this. Let me walk you through what we'll do.",
                    "I can see why this would be worrying. Let me provide you with clear information about how we'll handle this.",
                    "Your concerns are valid, and I want to help put your mind at ease. Here's what we can do..."
                ],
                'low': [
                    "I understand your concern. Let me provide you with some clarity on this.",
                    "I can help address your concerns. Here's what I can do for you.",
                    "Let me help clarify this situation for you."
                ]
            },
            'neutral': {
                'high': [
                    "Thank you for contacting us about this urgent matter. I'm going to prioritize your request and get back to you with a solution as quickly as possible.",
                    "I understand this is time-sensitive for you. Let me expedite this and provide you with an immediate response.",
                    "Thank you for reaching out. I can see this is urgent, so let me address this right away."
                ],
                'medium': [
                    "Thank you for your message. I'm happy to help you with this today.",
                    "I appreciate you contacting us. Let me look into this and provide you with a solution.",
                    "Thank you for reaching out. I'll be glad to assist you with this."
                ],
                'low': [
                    "Thank you for your inquiry. I'm here to help you with this.",
                    "I appreciate you contacting us. Let me assist you with your request.",
                    "Thank you for reaching out. I'll be happy to help you today."
                ]
            },
            'joy': {
                'high': [
                    "I'm so glad to hear from you! Your enthusiasm is wonderful, and I'm excited to help you with this.",
                    "It's fantastic to hear such positive energy! I'm thrilled to be able to assist you today.",
                    "Your excitement is contagious! I'm delighted to help you with this request."
                ],
                'medium': [
                    "It's great to hear from you! I'm happy to help you with this.",
                    "Thank you for your positive message. I'm pleased to assist you today.",
                    "I appreciate your kind words! I'm here to help you with whatever you need."
                ],
                'low': [
                    "Thank you for reaching out. I'm happy to help you with this.",
                    "I appreciate your message and I'm here to assist you.",
                    "Thank you for contacting us. I'll be glad to help you today."
                ]
            }
        }
        
        # De-escalation phrases
        self.de_escalation_phrases = [
            "I completely understand your frustration",
            "You're absolutely right to be concerned about this",
            "I can see why this would be upsetting",
            "Your feelings are completely valid",
            "I would feel the same way in your situation",
            "This is definitely not the experience we want for you",
            "I sincerely apologize for this inconvenience",
            "Let me take ownership of this issue",
            "I'm personally committed to resolving this for you"
        ]
        
        # Solution-oriented phrases
        self.solution_phrases = [
            "Here's what I'm going to do for you",
            "Let me work on a solution immediately",
            "I have a few options that might help",
            "Let's get this resolved right away",
            "I'm going to make sure this gets fixed",
            "Here's how we can move forward",
            "Let me take care of this for you",
            "I'll personally ensure this is resolved"
        ]
        
        # Closing phrases by escalation risk
        self.closing_phrases = {
            'high': [
                "I'll personally monitor this case and follow up with you within the hour. You have my commitment that we'll resolve this today.",
                "I'm escalating this to my supervisor immediately, and you'll hear back from us within 30 minutes. Thank you for your patience.",
                "I'm treating this as a priority case. I'll personally ensure you get a resolution today and follow up to make sure you're completely satisfied."
            ],
            'medium': [
                "I'll keep you updated on the progress and ensure this gets resolved quickly. Is there anything else I can help you with today?",
                "I'll follow up with you as soon as I have an update. Thank you for giving us the opportunity to make this right.",
                "I'll personally track this case to ensure it's resolved promptly. I appreciate your patience while we work on this."
            ],
            'low': [
                "Is there anything else I can help you with today? I'm here to ensure you have a great experience.",
                "Please don't hesitate to reach out if you have any other questions. I'm here to help!",
                "Thank you for contacting us. I hope this helps, and please let me know if you need anything else."
            ]
        }
        
        logger.info("Response generator initialized successfully")
    
    def generate_response(
        self,
        customer_message: str,
        sentiment_analysis: Dict[str, Any],
        escalation_prediction: Dict[str, Any],
        knowledge_context: str,
        source_documents: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate empathetic response based on analysis
        
        Args:
            customer_message: Original customer message
            sentiment_analysis: Results from sentiment analysis
            escalation_prediction: Results from escalation prediction
            knowledge_context: Retrieved knowledge base context
            source_documents: Source documents from RAG retrieval
            
        Returns:
            Dictionary containing generated response and metadata
        """
        
        # Extract key information
        dominant_emotion = self._get_dominant_emotion(sentiment_analysis.get('emotions', {}))
        urgency = sentiment_analysis.get('urgency', 'low')
        escalation_risk = escalation_prediction.get('risk_level', 'low')
        
        # Generate response using OpenAI if available, otherwise use templates
        if self.use_openai:
            response_text = self._generate_openai_response(
                customer_message, sentiment_analysis, escalation_prediction, knowledge_context
            )
        else:
            response_text = self._generate_template_response(
                dominant_emotion, urgency, escalation_risk, knowledge_context
            )
        
        # Apply tone calibration
        calibrated_response = self._calibrate_tone(
            response_text, sentiment_analysis, escalation_prediction
        )
        
        # Add solution and closing
        final_response = self._construct_final_response(
            calibrated_response, knowledge_context, escalation_risk, source_documents
        )
        
        # Generate metadata
        response_metadata = {
            'emotion_detected': dominant_emotion,
            'urgency_level': urgency,
            'escalation_risk': escalation_risk,
            'tone_adjustments': self._get_tone_adjustments(sentiment_analysis, escalation_prediction),
            'knowledge_sources_used': len(source_documents) if source_documents else 0,
            'response_strategy': self._get_response_strategy(escalation_risk),
            'generated_at': datetime.now().isoformat()
        }
        
        return {
            'response': final_response,
            'metadata': response_metadata
        }
    
    def _get_dominant_emotion(self, emotions: Dict[str, float]) -> str:
        """Get the dominant emotion from emotion scores"""
        if not emotions:
            return 'neutral'
        
        # Find emotion with highest score
        dominant_emotion = max(emotions, key=emotions.get)
        
        # Only consider it dominant if score is above threshold
        if emotions[dominant_emotion] > 0.3:
            return dominant_emotion
        else:
            return 'neutral'
    
    def _generate_openai_response(
        self,
        customer_message: str,
        sentiment_analysis: Dict[str, Any],
        escalation_prediction: Dict[str, Any],
        knowledge_context: str
    ) -> str:
        """Generate response using OpenAI GPT"""
        
        # Create context-aware prompt
        emotion = self._get_dominant_emotion(sentiment_analysis.get('emotions', {}))
        urgency = sentiment_analysis.get('urgency', 'low')
        escalation_risk = escalation_prediction.get('risk_level', 'low')
        
        prompt = f"""
You are an empathetic customer support representative. Generate a helpful, professional response to the customer's message.

Customer Message: "{customer_message}"

Context Information:
- Customer Emotion: {emotion}
- Urgency Level: {urgency}
- Escalation Risk: {escalation_risk}
- Relevant Knowledge: {knowledge_context[:1000]}

Instructions:
1. Be empathetic and acknowledge the customer's emotional state
2. Address their concerns directly using the knowledge context
3. Provide clear, actionable solutions
4. Match the tone to their urgency and emotion level
5. Use de-escalation techniques if escalation risk is high
6. Keep response professional but warm and human

Response:"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert customer support representative known for empathetic, solution-focused responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"OpenAI response generation failed: {e}")
            # Fall back to template-based response
            return self._generate_template_response(emotion, urgency, escalation_risk, knowledge_context)
    
    def _generate_template_response(
        self,
        emotion: str,
        urgency: str,
        escalation_risk: str,
        knowledge_context: str
    ) -> str:
        """Generate response using templates"""
        
        # Get appropriate opening based on emotion and urgency
        if emotion in self.response_templates:
            opening_templates = self.response_templates[emotion].get(urgency, self.response_templates[emotion]['low'])
        else:
            opening_templates = self.response_templates['neutral'].get(urgency, self.response_templates['neutral']['low'])
        
        opening = random.choice(opening_templates)
        
        # Add de-escalation if needed
        if escalation_risk in ['high', 'medium']:
            de_escalation = random.choice(self.de_escalation_phrases)
            opening = f"{de_escalation}. {opening}"
        
        return opening
    
    def _calibrate_tone(
        self,
        response: str,
        sentiment_analysis: Dict[str, Any],
        escalation_prediction: Dict[str, Any]
    ) -> str:
        """Calibrate response tone based on analysis"""
        
        escalation_risk = escalation_prediction.get('risk_level', 'low')
        emotions = sentiment_analysis.get('emotions', {})
        
        # Adjust for high escalation risk
        if escalation_risk == 'high':
            if not any(phrase in response.lower() for phrase in ['sincerely apologize', 'completely understand', 'personally ensure']):
                response = "I sincerely apologize for this situation. " + response
        
        # Adjust for high anger
        if emotions.get('anger', 0) > 0.6:
            if 'frustration' not in response.lower():
                response = response.replace("I understand", "I completely understand your frustration and")
        
        # Adjust for fear/anxiety
        if emotions.get('fear', 0) > 0.5:
            if 'reassure' not in response.lower():
                response += " I want to reassure you that we'll take care of this properly."
        
        return response
    
    def _construct_final_response(
        self,
        opening: str,
        knowledge_context: str,
        escalation_risk: str,
        source_documents: List[Dict[str, Any]] = None
    ) -> str:
        """Construct the final response with solution and closing"""
        
        # Extract key solution points from knowledge context
        solution_info = self._extract_solution_info(knowledge_context)
        
        # Add solution-oriented transition
        solution_intro = random.choice(self.solution_phrases)
        
        # Construct response parts
        response_parts = [opening]
        
        if solution_info:
            response_parts.extend([
                f"\n\n{solution_intro}:",
                solution_info
            ])
        
        # Add appropriate closing
        closing = random.choice(self.closing_phrases.get(escalation_risk, self.closing_phrases['low']))
        response_parts.append(f"\n\n{closing}")
        
        # Add source information if available
        if source_documents and len(source_documents) > 0:
            response_parts.append("\n\n📚 This information is based on our latest help articles and support documentation.")
        
        return "".join(response_parts)
    
    def _extract_solution_info(self, knowledge_context: str) -> str:
        """Extract actionable solution information from knowledge context"""
        if not knowledge_context:
            return "Let me look into the best solution for your specific situation."
        
        # Simple extraction of key solution steps
        lines = knowledge_context.split('\n')
        solution_lines = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['step', 'solution', 'resolve', 'fix', 'help']):
                if len(line) > 20 and len(line) < 200:  # Reasonable length
                    solution_lines.append(line)
        
        if solution_lines:
            return "\n".join(solution_lines[:3])  # Max 3 key points
        else:
            # Return first meaningful paragraph
            paragraphs = [p.strip() for p in knowledge_context.split('\n\n') if len(p.strip()) > 50]
            return paragraphs[0] if paragraphs else "I'll help you resolve this issue step by step."
    
    def _get_tone_adjustments(
        self,
        sentiment_analysis: Dict[str, Any],
        escalation_prediction: Dict[str, Any]
    ) -> List[str]:
        """Get list of tone adjustments made"""
        adjustments = []
        
        escalation_risk = escalation_prediction.get('risk_level', 'low')
        emotions = sentiment_analysis.get('emotions', {})
        
        if escalation_risk == 'high':
            adjustments.append('high_urgency_tone')
        
        if emotions.get('anger', 0) > 0.6:
            adjustments.append('de_escalation_language')
        
        if emotions.get('fear', 0) > 0.5:
            adjustments.append('reassuring_tone')
        
        if emotions.get('sadness', 0) > 0.5:
            adjustments.append('empathetic_language')
        
        if sentiment_analysis.get('urgency') == 'high':
            adjustments.append('expedited_response_tone')
        
        return adjustments
    
    def _get_response_strategy(self, escalation_risk: str) -> str:
        """Get response strategy based on escalation risk"""
        strategies = {
            'high': 'immediate_escalation_prevention',
            'medium': 'proactive_de_escalation',
            'low': 'standard_helpful_support'
        }
        return strategies.get(escalation_risk, 'standard_helpful_support')
    
    def get_response_suggestions(
        self,
        sentiment_analysis: Dict[str, Any],
        escalation_prediction: Dict[str, Any]
    ) -> List[str]:
        """Get response strategy suggestions for the agent"""
        
        suggestions = []
        
        escalation_risk = escalation_prediction.get('risk_level', 'low')
        emotions = sentiment_analysis.get('emotions', {})
        urgency = sentiment_analysis.get('urgency', 'low')
        
        # Escalation-based suggestions
        if escalation_risk == 'high':
            suggestions.extend([
                "🚨 Use immediate de-escalation language",
                "📞 Consider offering a phone call",
                "👥 Escalate to supervisor if needed",
                "⏰ Provide specific timeline for resolution"
            ])
        
        # Emotion-based suggestions
        if emotions.get('anger', 0) > 0.6:
            suggestions.extend([
                "😤 Acknowledge frustration explicitly",
                "🤝 Take ownership of the problem",
                "💪 Use confident, solution-focused language"
            ])
        
        if emotions.get('fear', 0) > 0.5:
            suggestions.extend([
                "🤗 Provide reassurance and clarity",
                "📋 Explain each step clearly",
                "🛡️ Emphasize security and reliability"
            ])
        
        if emotions.get('sadness', 0) > 0.5:
            suggestions.extend([
                "💙 Use empathetic, caring language",
                "🌟 Focus on positive outcomes",
                "🤲 Offer additional support if needed"
            ])
        
        # Urgency-based suggestions
        if urgency == 'high':
            suggestions.extend([
                "⚡ Respond quickly and efficiently",
                "🎯 Focus on immediate solutions",
                "📈 Prioritize this interaction"
            ])
        
        return suggestions 