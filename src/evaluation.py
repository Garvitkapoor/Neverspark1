"""
Evaluation Metrics for Customer Support RAG System
Comprehensive evaluation of retrieval accuracy, sentiment analysis, and response quality
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import json
from collections import defaultdict
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Comprehensive evaluation system for RAG performance
    """
    
    def __init__(self):
        """Initialize the evaluator"""
        self.evaluation_history = []
        
    def evaluate_retrieval_accuracy(
        self,
        queries: List[str],
        ground_truth_docs: List[List[str]],
        retrieved_docs: List[List[Dict[str, Any]]],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval accuracy using standard IR metrics
        
        Args:
            queries: List of search queries
            ground_truth_docs: List of relevant document IDs for each query
            retrieved_docs: List of retrieved documents for each query
            k_values: List of k values for precision@k and recall@k
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if len(queries) != len(ground_truth_docs) != len(retrieved_docs):
            raise ValueError("All input lists must have the same length")
        
        metrics = {f'precision_at_{k}': [] for k in k_values}
        metrics.update({f'recall_at_{k}': [] for k in k_values})
        metrics['mrr'] = []  # Mean Reciprocal Rank
        metrics['map'] = []  # Mean Average Precision
        
        for query, gt_docs, ret_docs in zip(queries, ground_truth_docs, retrieved_docs):
            # Extract document IDs from retrieved docs
            retrieved_ids = [doc.get('metadata', {}).get('id', doc.get('id', '')) 
                           for doc in ret_docs]
            
            # Calculate precision and recall at k
            for k in k_values:
                retrieved_k = retrieved_ids[:k]
                relevant_retrieved = set(retrieved_k) & set(gt_docs)
                
                precision_k = len(relevant_retrieved) / k if k > 0 else 0
                recall_k = len(relevant_retrieved) / len(gt_docs) if gt_docs else 0
                
                metrics[f'precision_at_{k}'].append(precision_k)
                metrics[f'recall_at_{k}'].append(recall_k)
            
            # Calculate MRR
            mrr = 0
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in gt_docs:
                    mrr = 1 / (i + 1)
                    break
            metrics['mrr'].append(mrr)
            
            # Calculate Average Precision
            relevant_positions = [i + 1 for i, doc_id in enumerate(retrieved_ids) 
                                if doc_id in gt_docs]
            if relevant_positions:
                ap = sum(len([p for p in relevant_positions if p <= pos]) / pos 
                        for pos in relevant_positions) / len(gt_docs)
            else:
                ap = 0
            metrics['map'].append(ap)
        
        # Calculate averages
        averaged_metrics = {}
        for metric, values in metrics.items():
            averaged_metrics[metric] = np.mean(values) if values else 0
        
        return averaged_metrics
    
    def evaluate_sentiment_accuracy(
        self,
        texts: List[str],
        ground_truth_sentiments: List[str],
        predicted_sentiments: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate sentiment analysis accuracy
        
        Args:
            texts: List of input texts
            ground_truth_sentiments: True sentiment labels
            predicted_sentiments: Predicted sentiment labels
            
        Returns:
            Dictionary containing sentiment evaluation metrics
        """
        if len(texts) != len(ground_truth_sentiments) != len(predicted_sentiments):
            raise ValueError("All input lists must have the same length")
        
        # Calculate accuracy
        correct_predictions = sum(1 for gt, pred in zip(ground_truth_sentiments, predicted_sentiments) 
                                if gt == pred)
        accuracy = correct_predictions / len(texts) if texts else 0
        
        # Calculate per-class metrics
        unique_sentiments = list(set(ground_truth_sentiments + predicted_sentiments))
        class_metrics = {}
        
        for sentiment in unique_sentiments:
            tp = sum(1 for gt, pred in zip(ground_truth_sentiments, predicted_sentiments) 
                    if gt == sentiment and pred == sentiment)
            fp = sum(1 for gt, pred in zip(ground_truth_sentiments, predicted_sentiments) 
                    if gt != sentiment and pred == sentiment)
            fn = sum(1 for gt, pred in zip(ground_truth_sentiments, predicted_sentiments) 
                    if gt == sentiment and pred != sentiment)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[sentiment] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Calculate macro averages
        macro_precision = np.mean([metrics['precision'] for metrics in class_metrics.values()])
        macro_recall = np.mean([metrics['recall'] for metrics in class_metrics.values()])
        macro_f1 = np.mean([metrics['f1'] for metrics in class_metrics.values()])
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'class_metrics': class_metrics
        }
    
    def evaluate_escalation_prediction(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth_escalations: List[bool]
    ) -> Dict[str, float]:
        """
        Evaluate escalation prediction performance
        
        Args:
            predictions: List of escalation prediction results
            ground_truth_escalations: True escalation outcomes
            
        Returns:
            Dictionary containing escalation prediction metrics
        """
        if len(predictions) != len(ground_truth_escalations):
            raise ValueError("Predictions and ground truth must have same length")
        
        # Convert predictions to binary (considering 'high' and 'medium' as escalation)
        predicted_escalations = [
            pred['risk_level'] in ['high', 'medium'] for pred in predictions
        ]
        
        # Calculate binary classification metrics
        tp = sum(1 for gt, pred in zip(ground_truth_escalations, predicted_escalations) 
                if gt and pred)
        fp = sum(1 for gt, pred in zip(ground_truth_escalations, predicted_escalations) 
                if not gt and pred)
        tn = sum(1 for gt, pred in zip(ground_truth_escalations, predicted_escalations) 
                if not gt and not pred)
        fn = sum(1 for gt, pred in zip(ground_truth_escalations, predicted_escalations) 
                if gt and not pred)
        
        accuracy = (tp + tn) / len(predictions) if predictions else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate AUC using probability scores
        probabilities = [pred['escalation_probability'] for pred in predictions]
        auc = self._calculate_auc(ground_truth_escalations, probabilities)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'auc': auc
        }
    
    def evaluate_response_quality(
        self,
        generated_responses: List[str],
        reference_responses: List[str],
        customer_messages: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate response quality using various metrics
        
        Args:
            generated_responses: AI-generated responses
            reference_responses: Human reference responses
            customer_messages: Original customer messages
            
        Returns:
            Dictionary containing response quality metrics
        """
        if len(generated_responses) != len(reference_responses) != len(customer_messages):
            raise ValueError("All input lists must have the same length")
        
        metrics = {}
        
        # BLEU-like score (simplified)
        bleu_scores = []
        for gen, ref in zip(generated_responses, reference_responses):
            bleu = self._calculate_simple_bleu(gen, ref)
            bleu_scores.append(bleu)
        metrics['bleu'] = np.mean(bleu_scores)
        
        # ROUGE-like score (simplified)
        rouge_scores = []
        for gen, ref in zip(generated_responses, reference_responses):
            rouge = self._calculate_simple_rouge(gen, ref)
            rouge_scores.append(rouge)
        metrics['rouge'] = np.mean(rouge_scores)
        
        # Response length analysis
        gen_lengths = [len(resp.split()) for resp in generated_responses]
        ref_lengths = [len(resp.split()) for resp in reference_responses]
        
        metrics['avg_response_length'] = np.mean(gen_lengths)
        metrics['length_ratio'] = np.mean(gen_lengths) / np.mean(ref_lengths) if ref_lengths else 0
        
        # Empathy score (keyword-based)
        empathy_scores = []
        empathy_keywords = [
            'understand', 'sorry', 'apologize', 'help', 'assist', 'support',
            'concern', 'appreciate', 'thank', 'resolve', 'solve'
        ]
        
        for response in generated_responses:
            response_lower = response.lower()
            empathy_score = sum(1 for keyword in empathy_keywords 
                              if keyword in response_lower)
            empathy_scores.append(empathy_score)
        
        metrics['avg_empathy_score'] = np.mean(empathy_scores)
        
        # Solution orientation score
        solution_scores = []
        solution_keywords = [
            'solution', 'resolve', 'fix', 'help', 'steps', 'try', 'check',
            'contact', 'support', 'assist', 'guide', 'process'
        ]
        
        for response in generated_responses:
            response_lower = response.lower()
            solution_score = sum(1 for keyword in solution_keywords 
                               if keyword in response_lower)
            solution_scores.append(solution_score)
        
        metrics['avg_solution_score'] = np.mean(solution_scores)
        
        return metrics
    
    def _calculate_simple_bleu(self, generated: str, reference: str, n: int = 2) -> float:
        """Calculate simplified BLEU score"""
        gen_words = generated.lower().split()
        ref_words = reference.lower().split()
        
        if not gen_words or not ref_words:
            return 0.0
        
        # Calculate n-gram precision
        gen_ngrams = self._get_ngrams(gen_words, n)
        ref_ngrams = self._get_ngrams(ref_words, n)
        
        if not gen_ngrams:
            return 0.0
        
        matches = sum(1 for ngram in gen_ngrams if ngram in ref_ngrams)
        precision = matches / len(gen_ngrams)
        
        # Apply brevity penalty
        bp = min(1.0, len(gen_words) / len(ref_words)) if ref_words else 0
        
        return bp * precision
    
    def _calculate_simple_rouge(self, generated: str, reference: str) -> float:
        """Calculate simplified ROUGE score"""
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        
        if not ref_words:
            return 0.0
        
        overlap = len(gen_words & ref_words)
        recall = overlap / len(ref_words)
        precision = overlap / len(gen_words) if gen_words else 0
        
        if recall + precision == 0:
            return 0.0
        
        f1 = 2 * (recall * precision) / (recall + precision)
        return f1
    
    def _get_ngrams(self, words: List[str], n: int) -> List[Tuple[str, ...]]:
        """Get n-grams from list of words"""
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    def _calculate_auc(self, y_true: List[bool], y_scores: List[float]) -> float:
        """Calculate AUC using trapezoidal rule"""
        if not y_true or not y_scores:
            return 0.0
        
        # Sort by scores (descending)
        sorted_pairs = sorted(zip(y_scores, y_true), reverse=True)
        
        # Calculate TPR and FPR at different thresholds
        thresholds = sorted(set(y_scores), reverse=True)
        tpr_fpr_pairs = []
        
        for threshold in thresholds:
            tp = sum(1 for score, label in sorted_pairs if score >= threshold and label)
            fp = sum(1 for score, label in sorted_pairs if score >= threshold and not label)
            tn = sum(1 for score, label in sorted_pairs if score < threshold and not label)
            fn = sum(1 for score, label in sorted_pairs if score < threshold and label)
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_fpr_pairs.append((fpr, tpr))
        
        # Add endpoints
        tpr_fpr_pairs = [(0, 0)] + tpr_fpr_pairs + [(1, 1)]
        tpr_fpr_pairs = sorted(set(tpr_fpr_pairs))
        
        # Calculate AUC using trapezoidal rule
        auc = 0
        for i in range(1, len(tpr_fpr_pairs)):
            x1, y1 = tpr_fpr_pairs[i-1]
            x2, y2 = tpr_fpr_pairs[i]
            auc += (x2 - x1) * (y1 + y2) / 2
        
        return auc
    
    def run_comprehensive_evaluation(
        self,
        rag_system,
        sentiment_analyzer,
        escalation_predictor,
        response_generator,
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of the entire system
        
        Args:
            rag_system: RAG system instance
            sentiment_analyzer: Sentiment analyzer instance
            escalation_predictor: Escalation predictor instance
            response_generator: Response generator instance
            test_data: Test dataset
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'retrieval_metrics': {},
            'sentiment_metrics': {},
            'escalation_metrics': {},
            'response_metrics': {},
            'overall_performance': {}
        }
        
        try:
            # Test retrieval accuracy
            if 'retrieval_test' in test_data:
                retrieval_test = test_data['retrieval_test']
                retrieved_docs = []
                
                for query in retrieval_test['queries']:
                    docs = rag_system.search(query, n_results=5)
                    retrieved_docs.append(docs)
                
                results['retrieval_metrics'] = self.evaluate_retrieval_accuracy(
                    retrieval_test['queries'],
                    retrieval_test['ground_truth_docs'],
                    retrieved_docs
                )
            
            # Test sentiment analysis
            if 'sentiment_test' in test_data:
                sentiment_test = test_data['sentiment_test']
                predicted_sentiments = []
                
                for text in sentiment_test['texts']:
                    analysis = sentiment_analyzer.analyze_sentiment(text)
                    predicted_sentiments.append(analysis['overall_sentiment'])
                
                results['sentiment_metrics'] = self.evaluate_sentiment_accuracy(
                    sentiment_test['texts'],
                    sentiment_test['ground_truth_sentiments'],
                    predicted_sentiments
                )
            
            # Test escalation prediction
            if 'escalation_test' in test_data:
                escalation_test = test_data['escalation_test']
                predictions = []
                
                for message in escalation_test['messages']:
                    pred = escalation_predictor.predict_escalation(
                        customer_id="test",
                        message=message
                    )
                    predictions.append(pred)
                
                results['escalation_metrics'] = self.evaluate_escalation_prediction(
                    predictions,
                    escalation_test['ground_truth_escalations']
                )
            
            # Test response generation
            if 'response_test' in test_data:
                response_test = test_data['response_test']
                generated_responses = []
                
                for i, message in enumerate(response_test['customer_messages']):
                    # Get sentiment and escalation for context
                    sentiment = sentiment_analyzer.analyze_sentiment(message)
                    escalation = escalation_predictor.predict_escalation("test", message)
                    context, sources = rag_system.get_context(message)
                    
                    response = response_generator.generate_response(
                        customer_message=message,
                        sentiment_analysis=sentiment,
                        escalation_prediction=escalation,
                        knowledge_context=context,
                        source_documents=sources
                    )
                    generated_responses.append(response['response'])
                
                results['response_metrics'] = self.evaluate_response_quality(
                    generated_responses,
                    response_test['reference_responses'],
                    response_test['customer_messages']
                )
            
            # Calculate overall performance score
            performance_scores = []
            
            if results['retrieval_metrics']:
                retrieval_score = np.mean([
                    results['retrieval_metrics'].get('precision_at_3', 0),
                    results['retrieval_metrics'].get('recall_at_3', 0),
                    results['retrieval_metrics'].get('mrr', 0)
                ])
                performance_scores.append(retrieval_score)
            
            if results['sentiment_metrics']:
                sentiment_score = results['sentiment_metrics'].get('accuracy', 0)
                performance_scores.append(sentiment_score)
            
            if results['escalation_metrics']:
                escalation_score = results['escalation_metrics'].get('f1', 0)
                performance_scores.append(escalation_score)
            
            if results['response_metrics']:
                response_score = np.mean([
                    results['response_metrics'].get('bleu', 0),
                    results['response_metrics'].get('rouge', 0),
                    min(results['response_metrics'].get('avg_empathy_score', 0) / 5, 1.0)
                ])
                performance_scores.append(response_score)
            
            results['overall_performance'] = {
                'overall_score': np.mean(performance_scores) if performance_scores else 0,
                'component_scores': {
                    'retrieval': performance_scores[0] if len(performance_scores) > 0 else 0,
                    'sentiment': performance_scores[1] if len(performance_scores) > 1 else 0,
                    'escalation': performance_scores[2] if len(performance_scores) > 2 else 0,
                    'response': performance_scores[3] if len(performance_scores) > 3 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            results['error'] = str(e)
        
        # Store evaluation results
        self.evaluation_history.append(results)
        
        return results
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report"""
        
        report = f"""
# Customer Support RAG System Evaluation Report

**Evaluation Date:** {results.get('timestamp', 'Unknown')}

## Overall Performance
- **Overall Score:** {results['overall_performance']['overall_score']:.3f}

### Component Scores:
- **Retrieval Performance:** {results['overall_performance']['component_scores']['retrieval']:.3f}
- **Sentiment Analysis:** {results['overall_performance']['component_scores']['sentiment']:.3f}
- **Escalation Prediction:** {results['overall_performance']['component_scores']['escalation']:.3f}
- **Response Generation:** {results['overall_performance']['component_scores']['response']:.3f}

## Detailed Metrics

### Retrieval Metrics
"""
        
        if results['retrieval_metrics']:
            for metric, value in results['retrieval_metrics'].items():
                report += f"- **{metric.replace('_', ' ').title()}:** {value:.3f}\n"
        
        report += "\n### Sentiment Analysis Metrics\n"
        if results['sentiment_metrics']:
            for metric, value in results['sentiment_metrics'].items():
                if metric != 'class_metrics':
                    report += f"- **{metric.replace('_', ' ').title()}:** {value:.3f}\n"
        
        report += "\n### Escalation Prediction Metrics\n"
        if results['escalation_metrics']:
            for metric, value in results['escalation_metrics'].items():
                report += f"- **{metric.replace('_', ' ').title()}:** {value:.3f}\n"
        
        report += "\n### Response Quality Metrics\n"
        if results['response_metrics']:
            for metric, value in results['response_metrics'].items():
                report += f"- **{metric.replace('_', ' ').title()}:** {value:.3f}\n"
        
        # Recommendations
        report += "\n## Recommendations\n"
        
        overall_score = results['overall_performance']['overall_score']
        if overall_score >= 0.8:
            report += "✅ **Excellent Performance** - System is performing very well across all components.\n"
        elif overall_score >= 0.6:
            report += "⚠️ **Good Performance** - System is performing well but has room for improvement.\n"
        else:
            report += "❌ **Needs Improvement** - System requires significant optimization.\n"
        
        # Component-specific recommendations
        component_scores = results['overall_performance']['component_scores']
        
        if component_scores['retrieval'] < 0.6:
            report += "- **Improve Retrieval:** Consider better embedding models or chunking strategies.\n"
        
        if component_scores['sentiment'] < 0.7:
            report += "- **Enhance Sentiment Analysis:** Fine-tune models or add more training data.\n"
        
        if component_scores['escalation'] < 0.7:
            report += "- **Refine Escalation Prediction:** Update escalation patterns and thresholds.\n"
        
        if component_scores['response'] < 0.6:
            report += "- **Improve Response Generation:** Enhance templates or fine-tune language models.\n"
        
        return report 