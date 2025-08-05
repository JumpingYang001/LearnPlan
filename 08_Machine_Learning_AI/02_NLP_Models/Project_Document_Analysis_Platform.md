# Project: Document Analysis Platform

*Duration: 4-6 weeks*  
*Difficulty: Intermediate to Advanced*  
*Tech Stack: Python, Transformers, Streamlit, PostgreSQL, Redis*

## Project Overview

The Document Analysis Platform is a comprehensive system that processes various document formats, extracts meaningful insights, and presents them through interactive visualizations. This platform combines multiple NLP techniques including document understanding, summarization, named entity recognition, sentiment analysis, and topic modeling.

### Business Value
- **Automated Document Processing**: Reduce manual document review time by 80%
- **Intelligent Information Extraction**: Extract key entities, dates, and relationships automatically
- **Scalable Analysis**: Process thousands of documents with consistent quality
- **Interactive Insights**: Provide stakeholders with visual dashboards for decision-making

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   API Gateway   │    │   File Storage  │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│   (AWS S3/Local)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Document Queue │    │  NLP Processing │    │   Vector Store  │
│    (Redis)      │◄──►│     Pipeline    │◄──►│   (Pinecone)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │    │   ML Models     │    │  Visualization  │
│  (PostgreSQL)   │    │  (Transformers) │    │    Engine       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Features Implementation

### 1. Document Understanding and Processing

#### Multi-Format Document Parser
```python
import PyPDF2
import docx
import pandas as pd
from pathlib import Path
import magic
import logging
from typing import Dict, Any, Optional

class DocumentProcessor:
    """Handles multiple document formats and extracts text content."""
    
    def __init__(self):
        self.supported_formats = {
            'application/pdf': self._extract_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._extract_docx,
            'text/plain': self._extract_txt,
            'application/vnd.ms-excel': self._extract_excel,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self._extract_excel
        }
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a document and extract metadata and content."""
        try:
            # Detect file type
            mime_type = magic.from_file(str(file_path), mime=True)
            
            if mime_type not in self.supported_formats:
                raise ValueError(f"Unsupported file type: {mime_type}")
            
            # Extract content
            content = self.supported_formats[mime_type](file_path)
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, mime_type)
            
            return {
                'file_path': str(file_path),
                'mime_type': mime_type,
                'content': content,
                'metadata': metadata,
                'word_count': len(content.split()),
                'char_count': len(content)
            }
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            raise
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX files."""
        doc = docx.Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    
    def _extract_txt(self, file_path: Path) -> str:
        """Extract text from plain text files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _extract_excel(self, file_path: Path) -> str:
        """Extract text from Excel files."""
        df = pd.read_excel(file_path)
        return df.to_string()
    
    def _extract_metadata(self, file_path: Path, mime_type: str) -> Dict[str, Any]:
        """Extract file metadata."""
        stat = file_path.stat()
        return {
            'filename': file_path.name,
            'size_bytes': stat.st_size,
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'mime_type': mime_type
        }
```

#### Advanced Text Preprocessing
```python
import re
import spacy
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    """Advanced text preprocessing for document analysis."""
    
    def __init__(self, language: str = 'en'):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text preprocessing pipeline."""
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Sentence segmentation
        sentences = sent_tokenize(cleaned_text)
        
        # Tokenization and linguistic analysis
        doc = self.nlp(cleaned_text)
        
        # Extract linguistic features
        tokens = [token.text for token in doc if not token.is_space]
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        # Named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'tokens': tokens,
            'lemmas': lemmas,
            'pos_tags': pos_tags,
            'entities': entities,
            'sentence_count': len(sentences),
            'token_count': len(tokens)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""''``]', '"', text)
        
        return text.strip()
```

### 2. Intelligent Summarization System

#### Multi-Strategy Summarization
```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

class IntelligentSummarizer:
    """Advanced document summarization with multiple strategies."""
    
    def __init__(self):
        # Load different models for different summarization strategies
        self.abstractive_model = pipeline(
            'summarization',
            model='facebook/bart-large-cnn',
            tokenizer='facebook/bart-large-cnn'
        )
        
        self.extractive_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
    def generate_summary(self, text: str, strategy: str = 'hybrid', 
                        max_length: int = 150) -> Dict[str, Any]:
        """Generate summary using specified strategy."""
        
        strategies = {
            'abstractive': self._abstractive_summary,
            'extractive': self._extractive_summary,
            'hybrid': self._hybrid_summary
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return strategies[strategy](text, max_length)
    
    def _abstractive_summary(self, text: str, max_length: int) -> Dict[str, Any]:
        """Generate abstractive summary using BART."""
        
        # Split long documents into chunks
        chunks = self._split_into_chunks(text, max_chunk_length=1024)
        
        summaries = []
        for chunk in chunks:
            try:
                summary = self.abstractive_model(
                    chunk,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                continue
        
        # Combine chunk summaries
        combined_summary = " ".join(summaries)
        
        # If combined summary is too long, summarize again
        if len(combined_summary.split()) > max_length:
            final_summary = self.abstractive_model(
                combined_summary,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
        else:
            final_summary = combined_summary
        
        return {
            'summary': final_summary,
            'strategy': 'abstractive',
            'confidence_score': self._calculate_confidence(text, final_summary),
            'compression_ratio': len(text.split()) / len(final_summary.split())
        }
    
    def _extractive_summary(self, text: str, max_length: int) -> Dict[str, Any]:
        """Generate extractive summary using TF-IDF and clustering."""
        
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 3:
            return {
                'summary': text,
                'strategy': 'extractive',
                'confidence_score': 1.0,
                'compression_ratio': 1.0
            }
        
        # Vectorize sentences
        tfidf_matrix = self.extractive_vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Select top sentences
        num_sentences = min(max_length // 20, len(sentences) // 3)
        top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_sentence_indices.sort()
        
        summary_sentences = [sentences[i] for i in top_sentence_indices]
        summary = " ".join(summary_sentences)
        
        return {
            'summary': summary,
            'strategy': 'extractive',
            'confidence_score': np.mean(sentence_scores[top_sentence_indices]),
            'compression_ratio': len(text.split()) / len(summary.split())
        }
    
    def _hybrid_summary(self, text: str, max_length: int) -> Dict[str, Any]:
        """Combine extractive and abstractive approaches."""
        
        # First, use extractive to identify key content
        extractive_result = self._extractive_summary(text, max_length * 2)
        key_content = extractive_result['summary']
        
        # Then, use abstractive to create final summary
        abstractive_result = self._abstractive_summary(key_content, max_length)
        
        return {
            'summary': abstractive_result['summary'],
            'strategy': 'hybrid',
            'confidence_score': (extractive_result['confidence_score'] + 
                               abstractive_result['confidence_score']) / 2,
            'compression_ratio': len(text.split()) / len(abstractive_result['summary'].split())
        }
    
    def _split_into_chunks(self, text: str, max_chunk_length: int = 1024) -> List[str]:
        """Split text into manageable chunks for processing."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > max_chunk_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _calculate_confidence(self, original: str, summary: str) -> float:
        """Calculate confidence score for summary quality."""
        # Simple heuristic based on key term overlap
        original_words = set(original.lower().split())
        summary_words = set(summary.lower().split())
        
        if not summary_words:
            return 0.0
        
        overlap = len(original_words.intersection(summary_words))
        return min(overlap / len(summary_words), 1.0)
```

### 3. Information Extraction Engine

#### Named Entity Recognition and Relationship Extraction
```python
import spacy
from spacy import displacy
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import re

class InformationExtractor:
    """Advanced information extraction from documents."""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
        # Custom entity patterns
        self.custom_patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'URL': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'CURRENCY': r'\$[0-9,]+\.?[0-9]*',
            'PERCENTAGE': r'\b[0-9]+\.?[0-9]*%\b'
        }
    
    def extract_information(self, text: str) -> Dict[str, Any]:
        """Comprehensive information extraction."""
        
        doc = self.nlp(text)
        
        # Named Entity Recognition
        entities = self._extract_entities(doc)
        
        # Custom pattern extraction
        custom_entities = self._extract_custom_patterns(text)
        
        # Relationship extraction
        relationships = self._extract_relationships(doc)
        
        # Key phrases extraction
        key_phrases = self._extract_key_phrases(doc)
        
        # Statistical analysis
        stats = self._calculate_text_statistics(doc)
        
        return {
            'entities': entities,
            'custom_entities': custom_entities,
            'relationships': relationships,
            'key_phrases': key_phrases,
            'statistics': stats,
            'processed_text': text
        }
    
    def _extract_entities(self, doc) -> Dict[str, List[Dict]]:
        """Extract named entities with context."""
        entities_by_type = defaultdict(list)
        
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_),
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'confidence': ent._.get('confidence', 0.9)  # Default confidence
            }
            
            entities_by_type[ent.label_].append(entity_info)
        
        # Count and rank entities
        entity_summary = {}
        for entity_type, entities in entities_by_type.items():
            entity_texts = [e['text'] for e in entities]
            entity_counts = Counter(entity_texts)
            
            entity_summary[entity_type] = {
                'count': len(entities),
                'unique_count': len(entity_counts),
                'most_common': entity_counts.most_common(5),
                'entities': entities
            }
        
        return entity_summary
    
    def _extract_custom_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using custom regex patterns."""
        custom_entities = {}
        
        for pattern_name, pattern in self.custom_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            custom_entities[pattern_name] = list(set(matches))  # Remove duplicates
        
        return custom_entities
    
    def _extract_relationships(self, doc) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        
        # Find entities that appear in the same sentence
        for sent in doc.sents:
            sent_entities = [ent for ent in sent.ents]
            
            # Find pairs of entities in the same sentence
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i+1:]:
                    # Extract the text between entities
                    start_pos = min(ent1.end, ent2.end)
                    end_pos = max(ent1.start, ent2.start)
                    
                    if start_pos < end_pos:
                        relation_text = doc[start_pos:end_pos].text
                    else:
                        relation_text = ""
                    
                    relationship = {
                        'entity1': {'text': ent1.text, 'label': ent1.label_},
                        'entity2': {'text': ent2.text, 'label': ent2.label_},
                        'relation_context': relation_text,
                        'sentence': sent.text,
                        'confidence': 0.7  # Heuristic confidence
                    }
                    
                    relationships.append(relationship)
        
        return relationships
    
    def _extract_key_phrases(self, doc) -> List[Dict[str, Any]]:
        """Extract key phrases using linguistic patterns."""
        key_phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Multi-word phrases
                key_phrases.append({
                    'text': chunk.text,
                    'type': 'noun_phrase',
                    'root': chunk.root.text,
                    'root_pos': chunk.root.pos_
                })
        
        # Extract verb phrases (simplified)
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                # Get verb and its direct objects
                verb_phrase = [token.text]
                for child in token.children:
                    if child.dep_ in ['dobj', 'pobj']:
                        verb_phrase.append(child.text)
                
                if len(verb_phrase) > 1:
                    key_phrases.append({
                        'text': ' '.join(verb_phrase),
                        'type': 'verb_phrase',
                        'root': token.text,
                        'root_pos': token.pos_
                    })
        
        return key_phrases
    
    def _calculate_text_statistics(self, doc) -> Dict[str, Any]:
        """Calculate various text statistics."""
        sentences = list(doc.sents)
        tokens = [token for token in doc if not token.is_space]
        words = [token for token in tokens if token.is_alpha]
        
        return {
            'sentence_count': len(sentences),
            'token_count': len(tokens),
            'word_count': len(words),
            'unique_words': len(set(token.lower_ for token in words)),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'lexical_diversity': len(set(token.lower_ for token in words)) / len(words) if words else 0,
            'pos_distribution': Counter(token.pos_ for token in tokens)
        }
```

### 4. Visualization and Dashboard

#### Interactive Dashboard with Streamlit
```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, Any, List

class DocumentAnalysisDashboard:
    """Interactive dashboard for document analysis results."""
    
    def __init__(self):
        st.set_page_config(page_title="Document Analysis Platform", layout="wide")
        
    def render_dashboard(self, analysis_results: Dict[str, Any]):
        """Render the complete dashboard."""
        
        st.title("📄 Document Analysis Platform")
        st.markdown("---")
        
        # Sidebar for navigation
        analysis_type = st.sidebar.selectbox(
            "Select Analysis View",
            ["Overview", "Entity Analysis", "Summary", "Relationships", "Statistics"]
        )
        
        if analysis_type == "Overview":
            self._render_overview(analysis_results)
        elif analysis_type == "Entity Analysis":
            self._render_entity_analysis(analysis_results)
        elif analysis_type == "Summary":
            self._render_summary_analysis(analysis_results)
        elif analysis_type == "Relationships":
            self._render_relationship_analysis(analysis_results)
        elif analysis_type == "Statistics":
            self._render_statistical_analysis(analysis_results)
    
    def _render_overview(self, results: Dict[str, Any]):
        """Render overview dashboard."""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents Processed", results.get('document_count', 0))
        
        with col2:
            st.metric("Total Words", results.get('total_words', 0))
        
        with col3:
            st.metric("Entities Extracted", results.get('entity_count', 0))
        
        with col4:
            st.metric("Average Confidence", f"{results.get('avg_confidence', 0):.2f}")
        
        # Document processing timeline
        if 'processing_timeline' in results:
            fig = px.line(
                results['processing_timeline'],
                x='timestamp',
                y='documents_processed',
                title='Document Processing Timeline'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Word cloud
        if 'text_content' in results:
            st.subheader("Word Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
                results['text_content']
            )
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    def _render_entity_analysis(self, results: Dict[str, Any]):
        """Render entity analysis dashboard."""
        
        st.header("🏷️ Entity Analysis")
        
        entities = results.get('entities', {})
        
        if not entities:
            st.warning("No entities found in the analysis results.")
            return
        
        # Entity distribution
        entity_counts = {entity_type: data['count'] for entity_type, data in entities.items()}
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=list(entity_counts.keys()),
                y=list(entity_counts.values()),
                title="Entity Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                values=list(entity_counts.values()),
                names=list(entity_counts.keys()),
                title="Entity Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed entity exploration
        st.subheader("Entity Details")
        
        selected_entity_type = st.selectbox(
            "Select entity type to explore:",
            list(entities.keys())
        )
        
        if selected_entity_type in entities:
            entity_data = entities[selected_entity_type]
            
            # Most common entities
            st.write(f"**Most Common {selected_entity_type} Entities:**")
            most_common_df = pd.DataFrame(
                entity_data['most_common'],
                columns=['Entity', 'Count']
            )
            st.dataframe(most_common_df)
            
            # All entities with context
            st.write(f"**All {selected_entity_type} Entities:**")
            entity_details_df = pd.DataFrame(entity_data['entities'])
            st.dataframe(entity_details_df)
    
    def _render_summary_analysis(self, results: Dict[str, Any]):
        """Render summary analysis dashboard."""
        
        st.header("📝 Summary Analysis")
        
        summary_data = results.get('summary', {})
        
        if not summary_data:
            st.warning("No summary data available.")
            return
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Compression Ratio", f"{summary_data.get('compression_ratio', 0):.2f}x")
        
        with col2:
            st.metric("Strategy Used", summary_data.get('strategy', 'Unknown'))
        
        with col3:
            st.metric("Confidence Score", f"{summary_data.get('confidence_score', 0):.2f}")
        
        # Summary text
        st.subheader("Generated Summary")
        st.write(summary_data.get('summary', 'No summary available'))
        
        # Summary comparison (if multiple strategies were used)
        if 'strategy_comparison' in summary_data:
            st.subheader("Strategy Comparison")
            
            comparison_df = pd.DataFrame(summary_data['strategy_comparison'])
            
            fig = px.bar(
                comparison_df,
                x='strategy',
                y='confidence_score',
                title='Summary Strategy Performance'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_relationship_analysis(self, results: Dict[str, Any]):
        """Render relationship analysis dashboard."""
        
        st.header("🔗 Relationship Analysis")
        
        relationships = results.get('relationships', [])
        
        if not relationships:
            st.warning("No relationships found in the analysis results.")
            return
        
        # Relationship network visualization
        st.subheader("Entity Relationship Network")
        
        # Create network graph
        G = nx.Graph()
        
        for rel in relationships:
            entity1 = rel['entity1']['text']
            entity2 = rel['entity2']['text']
            G.add_edge(entity1, entity2, weight=rel['confidence'])
        
        # Generate positions for nodes
        pos = nx.spring_layout(G)
        
        # Create Plotly network visualization
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x, node_y = [], []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    xanchor="left",
                    titleside="right"
                )
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Entity Relationship Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Relationships between extracted entities",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Relationship details table
        st.subheader("Relationship Details")
        relationships_df = pd.DataFrame([
            {
                'Entity 1': rel['entity1']['text'],
                'Entity 1 Type': rel['entity1']['label'],
                'Entity 2': rel['entity2']['text'],
                'Entity 2 Type': rel['entity2']['label'],
                'Context': rel['relation_context'],
                'Confidence': rel['confidence']
            }
            for rel in relationships
        ])
        
        st.dataframe(relationships_df)
    
    def _render_statistical_analysis(self, results: Dict[str, Any]):
        """Render statistical analysis dashboard."""
        
        st.header("📊 Statistical Analysis")
        
        stats = results.get('statistics', {})
        
        if not stats:
            st.warning("No statistical data available.")
            return
        
        # Text statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sentences", stats.get('sentence_count', 0))
        
        with col2:
            st.metric("Words", stats.get('word_count', 0))
        
        with col3:
            st.metric("Unique Words", stats.get('unique_words', 0))
        
        with col4:
            st.metric("Lexical Diversity", f"{stats.get('lexical_diversity', 0):.3f}")
        
        # POS distribution
        if 'pos_distribution' in stats:
            st.subheader("Part-of-Speech Distribution")
            
            pos_df = pd.DataFrame(
                list(stats['pos_distribution'].items()),
                columns=['POS Tag', 'Count']
            )
            
            fig = px.bar(pos_df, x='POS Tag', y='Count', title='Part-of-Speech Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Reading complexity analysis
        st.subheader("Reading Complexity")
        
        complexity_metrics = {
            'Average Sentence Length': stats.get('avg_sentence_length', 0),
            'Lexical Diversity': stats.get('lexical_diversity', 0),
            'Unique Word Ratio': stats.get('unique_words', 0) / stats.get('word_count', 1)
        }
        
        complexity_df = pd.DataFrame(
            list(complexity_metrics.items()),
            columns=['Metric', 'Value']
        )
        
        fig = px.bar(complexity_df, x='Metric', y='Value', title='Text Complexity Metrics')
        st.plotly_chart(fig, use_container_width=True)
```

### 5. Complete Application Integration

#### Main Application Controller
```python
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

class DocumentAnalysisPlatform:
    """Main platform controller integrating all components."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.preprocessor = TextPreprocessor()
        self.summarizer = IntelligentSummarizer()
        self.extractor = InformationExtractor()
        self.dashboard = DocumentAnalysisDashboard()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def analyze_documents(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Analyze multiple documents and return comprehensive results."""
        
        self.logger.info(f"Starting analysis of {len(file_paths)} documents")
        
        all_results = {
            'documents': [],
            'aggregate_stats': {},
            'processing_timeline': [],
            'errors': []
        }
        
        for file_path in file_paths:
            try:
                start_time = datetime.now()
                
                # Process document
                doc_data = self.processor.process_document(file_path)
                
                # Preprocess text
                preprocessed = self.preprocessor.preprocess_text(doc_data['content'])
                
                # Generate summary
                summary = self.summarizer.generate_summary(
                    doc_data['content'], 
                    strategy='hybrid'
                )
                
                # Extract information
                extracted_info = self.extractor.extract_information(doc_data['content'])
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Compile results
                doc_results = {
                    'file_info': doc_data,
                    'preprocessing': preprocessed,
                    'summary': summary,
                    'extracted_info': extracted_info,
                    'processing_time': processing_time,
                    'timestamp': end_time
                }
                
                all_results['documents'].append(doc_results)
                
                # Update timeline
                all_results['processing_timeline'].append({
                    'timestamp': end_time,
                    'documents_processed': len(all_results['documents']),
                    'processing_time': processing_time
                })
                
                self.logger.info(f"Completed analysis of {file_path.name} in {processing_time:.2f}s")
                
            except Exception as e:
                error_info = {
                    'file_path': str(file_path),
                    'error': str(e),
                    'timestamp': datetime.now()
                }
                all_results['errors'].append(error_info)
                self.logger.error(f"Error processing {file_path}: {e}")
        
        # Calculate aggregate statistics
        all_results['aggregate_stats'] = self._calculate_aggregate_stats(all_results['documents'])
        
        return all_results
    
    def _calculate_aggregate_stats(self, document_results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate statistics across all documents."""
        
        if not document_results:
            return {}
        
        total_words = sum(doc['file_info']['word_count'] for doc in document_results)
        total_entities = sum(
            sum(data['count'] for data in doc['extracted_info']['entities'].values())
            for doc in document_results
        )
        
        avg_confidence = sum(
            doc['summary']['confidence_score'] for doc in document_results
        ) / len(document_results)
        
        # Aggregate entity types
        all_entities = {}
        for doc in document_results:
            for entity_type, data in doc['extracted_info']['entities'].items():
                if entity_type not in all_entities:
                    all_entities[entity_type] = {'count': 0, 'entities': []}
                all_entities[entity_type]['count'] += data['count']
                all_entities[entity_type]['entities'].extend(data['entities'])
        
        return {
            'document_count': len(document_results),
            'total_words': total_words,
            'entity_count': total_entities,
            'avg_confidence': avg_confidence,
            'entities': all_entities,
            'processing_times': [doc['processing_time'] for doc in document_results]
        }
    
    def launch_dashboard(self, analysis_results: Dict[str, Any]):
        """Launch the interactive dashboard."""
        self.dashboard.render_dashboard(analysis_results)

# Usage example
if __name__ == "__main__":
    platform = DocumentAnalysisPlatform()
    
    # Example usage
    documents = [
        Path("sample_document1.pdf"),
        Path("sample_document2.docx"),
        Path("sample_document3.txt")
    ]
    
    # Run analysis
    results = asyncio.run(platform.analyze_documents(documents))
    
    # Launch dashboard
    platform.launch_dashboard(results)
```

## Deployment and Production Considerations

### Docker Configuration
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Requirements.txt
```txt
streamlit==1.28.0
transformers==4.30.0
torch==2.0.0
spacy==3.6.0
pandas==2.0.3
plotly==5.15.0
scikit-learn==1.3.0
PyPDF2==3.0.1
python-docx==0.8.11
python-magic==0.4.27
wordcloud==1.9.2
networkx==3.1
nltk==3.8.1
redis==4.6.0
psycopg2-binary==2.9.7
fastapi==0.100.0
uvicorn==0.23.0
```

This comprehensive Document Analysis Platform provides:

✅ **Multi-format document processing** (PDF, DOCX, TXT, Excel)  
✅ **Advanced text preprocessing** with linguistic analysis  
✅ **Intelligent summarization** using multiple strategies  
✅ **Comprehensive information extraction** (entities, relationships, key phrases)  
✅ **Interactive visualization dashboard** with multiple views  
✅ **Scalable architecture** for production deployment  
✅ **Error handling and logging** for robust operation  
✅ **Docker support** for easy deployment
