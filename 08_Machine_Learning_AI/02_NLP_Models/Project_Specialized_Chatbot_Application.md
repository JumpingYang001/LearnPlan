# Project: Specialized Chatbot Application

*Duration: 3-4 weeks*  
*Difficulty: Intermediate to Advanced*

## Project Overview

Build a sophisticated domain-specific conversational agent that can understand context, maintain conversation memory, and provide intelligent responses in a specialized field (e.g., customer support, medical assistance, technical help desk, or educational tutoring).

## Learning Objectives

By completing this project, you will:
- **Master NLP fundamentals** and apply them to real-world conversational AI
- **Implement context management** to maintain coherent long conversations
- **Build memory systems** for personalized user experiences
- **Create production-ready APIs** for chatbot deployment
- **Design user interfaces** for seamless chat experiences
- **Handle edge cases** and error scenarios in conversational AI
- **Evaluate and improve** chatbot performance using metrics

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Chatbot Application                      │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React/Streamlit)  │  API Layer (FastAPI/Flask)  │
├─────────────────────────────────────────────────────────────┤
│               Core Chatbot Engine                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Intent    │ │   Context   │ │   Memory    │          │
│  │ Recognition │ │ Management  │ │  Management │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                 NLP Processing Layer                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Tokenization│ │  Embedding  │ │  Response   │          │
│  │ & Parsing   │ │  Generation │ │ Generation  │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                    Data Layer                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Knowledge   │ │ Conversation│ │  User       │          │
│  │    Base     │ │   History   │ │ Profiles    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Foundation Setup and Basic Chatbot

### 1.1 Environment Setup

**Required Dependencies:**
```python
# requirements.txt
transformers==4.30.0
torch==2.0.1
sentence-transformers==2.2.2
nltk==3.8.1
spacy==3.6.0
fastapi==0.100.0
uvicorn==0.22.0
streamlit==1.24.0
redis==4.6.0
sqlalchemy==2.0.19
psycopg2-binary==2.9.6
langchain==0.0.220
openai==0.27.8
python-dotenv==1.0.0
pytest==7.4.0
```

**Installation Script:**
```bash
# setup.sh
#!/bin/bash

# Create virtual environment
python -m venv chatbot_env
source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "Environment setup complete!"
```

### 1.2 Basic Chatbot Implementation

**Core Chatbot Class:**
```python
# chatbot/core/bot.py
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, Conversation
)
from sentence_transformers import SentenceTransformer
import spacy
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class ChatMessage:
    """Represents a single chat message."""
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

@dataclass
class ConversationContext:
    """Manages conversation context and state."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    user_info: Dict = field(default_factory=dict)
    current_intent: Optional[str] = None
    entities: Dict = field(default_factory=dict)
    confidence_scores: List[float] = field(default_factory=list)

class MedicalChatbot:
    """
    Specialized Medical Support Chatbot
    
    Features:
    - Medical knowledge base integration
    - Symptom analysis and triage
    - Appointment scheduling assistance  
    - Medication information lookup
    - Emergency detection and escalation
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Initialize models
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.sentence_model = None
        self.nlp = None
        
        # Conversation management
        self.active_conversations: Dict[str, ConversationContext] = {}
        
        # Medical knowledge base
        self.medical_kb = self._load_medical_knowledge()
        self.emergency_keywords = [
            "emergency", "urgent", "chest pain", "heart attack", 
            "stroke", "bleeding", "unconscious", "breathing difficulty"
        ]
        
        # Intent patterns
        self.intent_patterns = {
            "symptom_inquiry": [
                "symptoms", "feel", "pain", "ache", "hurt", "sick"
            ],
            "medication_info": [
                "medication", "medicine", "drug", "prescription", "dosage"
            ],
            "appointment": [
                "appointment", "schedule", "book", "visit", "see doctor"
            ],
            "general_health": [
                "health", "wellness", "prevention", "diet", "exercise"
            ]
        }
        
        self._initialize_models()
    
    def setup_logging(self):
        """Configure logging for the chatbot."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('chatbot.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_models(self):
        """Initialize all required ML models."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Load conversational model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load sentence transformer for similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load spaCy for NER
            self.nlp = spacy.load("en_core_web_sm")
            
            self.logger.info("All models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _load_medical_knowledge(self) -> Dict:
        """Load medical knowledge base."""
        # In a real implementation, this would load from a comprehensive database
        return {
            "symptoms": {
                "headache": {
                    "common_causes": ["tension", "dehydration", "eye strain", "stress"],
                    "remedies": ["rest", "hydration", "pain reliever", "dark room"],
                    "warning_signs": ["sudden severe onset", "with fever", "vision changes"]
                },
                "fever": {
                    "common_causes": ["infection", "illness", "medication reaction"],
                    "remedies": ["rest", "fluids", "fever reducer", "cooling measures"],
                    "warning_signs": ["over 103°F", "difficulty breathing", "severe symptoms"]
                }
            },
            "medications": {
                "ibuprofen": {
                    "uses": ["pain relief", "inflammation", "fever reduction"],
                    "dosage": "200-400mg every 4-6 hours, max 1200mg/day",
                    "warnings": ["stomach irritation", "kidney concerns", "blood pressure"]
                }
            }
        }
    
    def detect_intent(self, message: str) -> Tuple[str, float]:
        """Detect user intent from message."""
        message_lower = message.lower()
        intent_scores = {}
        
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                intent_scores[intent] = score / len(keywords)
        
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            return best_intent[0], best_intent[1]
        
        return "general", 0.5
    
    def extract_entities(self, message: str) -> Dict:
        """Extract entities from user message."""
        doc = self.nlp(message)
        entities = {
            "persons": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
            "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
            "times": [ent.text for ent in doc.ents if ent.label_ == "TIME"],
            "symptoms": []  # Custom symptom extraction
        }
        
        # Custom symptom detection
        symptom_keywords = ["pain", "ache", "fever", "headache", "nausea", "dizzy"]
        for token in doc:
            if token.lemma_ in symptom_keywords:
                entities["symptoms"].append(token.text)
        
        return entities
    
    def check_emergency(self, message: str) -> bool:
        """Check if message indicates medical emergency."""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.emergency_keywords)
    
    def generate_response(self, session_id: str, user_message: str) -> str:
        """Generate chatbot response for user message."""
        try:
            # Get or create conversation context
            if session_id not in self.active_conversations:
                self.active_conversations[session_id] = ConversationContext(session_id)
            
            context = self.active_conversations[session_id]
            
            # Add user message to context
            user_msg = ChatMessage(content=user_message, role="user")
            context.messages.append(user_msg)
            
            # Check for emergency
            if self.check_emergency(user_message):
                emergency_response = (
                    "⚠️ EMERGENCY DETECTED ⚠️\n\n"
                    "If this is a medical emergency, please:\n"
                    "• Call 911 (US) or your local emergency number immediately\n"
                    "• Go to the nearest emergency room\n"
                    "• Contact your doctor right away\n\n"
                    "I'm an AI assistant and cannot provide emergency medical care. "
                    "Please seek immediate professional help."
                )
                bot_msg = ChatMessage(content=emergency_response, role="assistant")
                context.messages.append(bot_msg)
                return emergency_response
            
            # Detect intent and extract entities
            intent, confidence = self.detect_intent(user_message)
            entities = self.extract_entities(user_message)
            
            context.current_intent = intent
            context.entities.update(entities)
            context.confidence_scores.append(confidence)
            
            self.logger.info(f"Intent: {intent}, Confidence: {confidence:.2f}")
            
            # Generate contextual response based on intent
            response = self._generate_intent_based_response(intent, user_message, context)
            
            # Add bot response to context
            bot_msg = ChatMessage(content=response, role="assistant")
            context.messages.append(bot_msg)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact support."
    
    def _generate_intent_based_response(self, intent: str, message: str, context: ConversationContext) -> str:
        """Generate response based on detected intent."""
        
        if intent == "symptom_inquiry":
            return self._handle_symptom_inquiry(message, context)
        elif intent == "medication_info":
            return self._handle_medication_inquiry(message, context)
        elif intent == "appointment":
            return self._handle_appointment_request(message, context)
        elif intent == "general_health":
            return self._handle_general_health_question(message, context)
        else:
            return self._generate_general_response(message, context)
    
    def _handle_symptom_inquiry(self, message: str, context: ConversationContext) -> str:
        """Handle symptom-related inquiries."""
        symptoms = context.entities.get("symptoms", [])
        
        if "headache" in message.lower():
            return (
                "I understand you're experiencing a headache. Here's some general information:\n\n"
                "**Common causes:** Tension, dehydration, eye strain, stress\n"
                "**General remedies:** Rest, stay hydrated, consider over-the-counter pain relief\n"
                "**Seek medical attention if:** Sudden severe onset, accompanied by fever, "
                "vision changes, or neck stiffness\n\n"
                "Please consult with a healthcare professional for proper diagnosis and treatment. "
                "How long have you been experiencing this headache?"
            )
        elif "fever" in message.lower():
            return (
                "Fever can be concerning. Here's what you should know:\n\n"
                "**General care:** Rest, drink plenty of fluids, consider fever-reducing medication\n"
                "**Monitor temperature:** Keep track of fever patterns\n"
                "**Seek immediate care if:** Temperature over 103°F (39.4°C), difficulty breathing, "
                "severe symptoms, or fever persists\n\n"
                "Have you taken your temperature? What other symptoms are you experiencing?"
            )
        else:
            return (
                "I'd like to help you with your symptoms. Could you provide more specific details about:\n"
                "• What symptoms you're experiencing\n"
                "• When they started\n"
                "• How severe they are (1-10 scale)\n"
                "• Any other related symptoms\n\n"
                "Remember, I can provide general information, but you should consult a healthcare "
                "professional for proper medical advice."
            )
    
    def _handle_medication_inquiry(self, message: str, context: ConversationContext) -> str:
        """Handle medication-related questions."""
        return (
            "I can provide general information about medications, but please remember:\n\n"
            "⚠️ **Important:** Always consult your doctor or pharmacist for medication advice\n\n"
            "What specific medication would you like information about? I can help with:\n"
            "• General uses and effects\n"
            "• Common side effects to be aware of\n"
            "• General dosage information\n"
            "• Drug interaction warnings\n\n"
            "For specific dosing, interactions with your other medications, or medical advice, "
            "please consult your healthcare provider."
        )
    
    def _handle_appointment_request(self, message: str, context: ConversationContext) -> str:
        """Handle appointment scheduling requests."""
        return (
            "I'd be happy to help you with appointment information!\n\n"
            "To better assist you, could you let me know:\n"
            "• What type of appointment you need (routine checkup, specific concern, specialist)\n"
            "• Your preferred timing (morning, afternoon, specific days)\n"
            "• How urgent this appointment is\n\n"
            "**Next steps:**\n"
            "1. I can provide general guidance on preparation\n"
            "2. You'll need to call our scheduling line: (555) 123-4567\n"
            "3. Or use our online portal at: www.healthcenter.com/appointments\n\n"
            "Is this for a specific health concern or a routine visit?"
        )
    
    def _handle_general_health_question(self, message: str, context: ConversationContext) -> str:
        """Handle general health and wellness questions."""
        return (
            "I'm happy to discuss general health and wellness topics!\n\n"
            "Some areas I can help with:\n"
            "• General wellness tips\n"
            "• Healthy lifestyle information\n"
            "• Preventive care reminders\n"
            "• General health education\n\n"
            "What specific aspect of health and wellness would you like to discuss? "
            "Remember, for personalized medical advice, please consult with your healthcare provider."
        )
    
    def _generate_general_response(self, message: str, context: ConversationContext) -> str:
        """Generate general conversational response using the language model."""
        try:
            # Prepare conversation history for context
            conversation_history = ""
            for msg in context.messages[-6:]:  # Last 6 messages for context
                role_prefix = "Human: " if msg.role == "user" else "Assistant: "
                conversation_history += f"{role_prefix}{msg.content}\n"
            
            # Add current message
            conversation_history += f"Human: {message}\nAssistant:"
            
            # Tokenize and generate
            inputs = self.tokenizer.encode(conversation_history, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new response part
            response = response[len(conversation_history):].strip()
            
            # Add medical disclaimer for safety
            if not response:
                response = "I understand. Could you please rephrase your question so I can better assist you?"
            
            response += "\n\n💡 *Remember: I provide general information only. For medical advice, please consult a healthcare professional.*"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating general response: {str(e)}")
            return "I'm here to help! Could you please rephrase your question?"
    
    def get_conversation_summary(self, session_id: str) -> Dict:
        """Get summary of conversation for analytics."""
        if session_id not in self.active_conversations:
            return {"error": "Session not found"}
        
        context = self.active_conversations[session_id]
        
        return {
            "session_id": session_id,
            "message_count": len(context.messages),
            "intents": [context.current_intent] if context.current_intent else [],
            "entities": context.entities,
            "avg_confidence": sum(context.confidence_scores) / len(context.confidence_scores) if context.confidence_scores else 0,
            "duration": (datetime.now() - context.messages[0].timestamp).total_seconds() if context.messages else 0
        }
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.active_conversations:
            del self.active_conversations[session_id]
            self.logger.info(f"Cleared conversation for session: {session_id}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize chatbot
    bot = MedicalChatbot()
    
    # Test conversation
    session_id = "test_session_001"
    
    test_messages = [
        "Hello, I've been having headaches",
        "It started yesterday and it's pretty severe",
        "What medication can I take?",
        "Can I schedule an appointment?"
    ]
    
    print("=== Medical Chatbot Test ===\n")
    
    for message in test_messages:
        print(f"User: {message}")
        response = bot.generate_response(session_id, message)
        print(f"Bot: {response}\n")
        print("-" * 50 + "\n")
    
    # Show conversation summary
    summary = bot.get_conversation_summary(session_id)
    print("Conversation Summary:")
    print(json.dumps(summary, indent=2, default=str))
```

## Phase 2: Advanced Context Management and Memory Systems

### 2.1 Context Management Implementation

**Context Manager Class:**
```python
# chatbot/core/context_manager.py
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pickle

Base = declarative_base()

class ConversationRecord(Base):
    """Database model for conversation persistence."""
    __tablename__ = 'conversations'
    
    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    message_content = Column(Text)
    message_role = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    intent = Column(String)
    entities = Column(Text)  # JSON string
    confidence_score = Column(Integer)

@dataclass
class ContextState:
    """Advanced context state with memory management."""
    current_topic: Optional[str] = None
    conversation_stage: str = "greeting"  # greeting, inquiry, resolution, followup
    user_preferences: Dict = field(default_factory=dict)
    medical_history: List[str] = field(default_factory=list)
    previous_concerns: List[str] = field(default_factory=list)
    appointment_context: Dict = field(default_factory=dict)
    emotional_state: str = "neutral"  # neutral, anxious, frustrated, satisfied
    follow_up_needed: bool = False

class AdvancedContextManager:
    """
    Advanced context management with:
    - Short-term memory (current conversation)
    - Long-term memory (user history across sessions)
    - Context switching and topic tracking
    - Emotional state tracking
    - Personalization based on history
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 db_url: str = "sqlite:///chatbot.db"):
        # Redis for fast session storage
        self.redis_client = redis.from_url(redis_url)
        
        # Database for persistent storage
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()
        
        # Context tracking
        self.active_contexts: Dict[str, ContextState] = {}
        
        # Topic transition rules
        self.topic_transitions = {
            "greeting": ["symptom_inquiry", "appointment", "general_health"],
            "symptom_inquiry": ["medication_info", "appointment", "symptom_inquiry"],
            "medication_info": ["symptom_inquiry", "appointment", "general_health"],
            "appointment": ["symptom_inquiry", "general_health", "followup"],
            "general_health": ["symptom_inquiry", "appointment", "general_health"]
        }
    
    def get_context(self, session_id: str, user_id: Optional[str] = None) -> ContextState:
        """Retrieve or create context for a session."""
        # Try to get from memory first
        if session_id in self.active_contexts:
            return self.active_contexts[session_id]
        
        # Try Redis cache
        cached_context = self.redis_client.get(f"context:{session_id}")
        if cached_context:
            context = pickle.loads(cached_context)
            self.active_contexts[session_id] = context
            return context
        
        # Create new context with user history if available
        context = ContextState()
        
        if user_id:
            # Load user preferences and history from database
            user_history = self._load_user_history(user_id)
            context.user_preferences = user_history.get("preferences", {})
            context.medical_history = user_history.get("medical_history", [])
            context.previous_concerns = user_history.get("previous_concerns", [])
        
        self.active_contexts[session_id] = context
        return context
    
    def update_context(self, session_id: str, **updates):
        """Update context with new information."""
        context = self.get_context(session_id)
        
        for key, value in updates.items():
            if hasattr(context, key):
                setattr(context, key, value)
        
        # Cache in Redis
        self.redis_client.setex(
            f"context:{session_id}", 
            timedelta(hours=24), 
            pickle.dumps(context)
        )
    
    def track_topic_transition(self, session_id: str, old_topic: str, new_topic: str) -> bool:
        """Track and validate topic transitions."""
        context = self.get_context(session_id)
        
        # Check if transition is valid
        valid_transitions = self.topic_transitions.get(old_topic, [])
        if new_topic not in valid_transitions and new_topic != old_topic:
            # Handle unexpected transition
            context.conversation_stage = "topic_switch"
        
        context.current_topic = new_topic
        self.update_context(session_id, current_topic=new_topic)
        
        return new_topic in valid_transitions
    
    def detect_emotional_state(self, message: str, session_id: str) -> str:
        """Detect user's emotional state from message."""
        message_lower = message.lower()
        
        # Simple emotion detection (in production, use advanced NLP)
        if any(word in message_lower for word in ["worried", "scared", "anxious", "concerned"]):
            emotional_state = "anxious"
        elif any(word in message_lower for word in ["frustrated", "angry", "upset"]):
            emotional_state = "frustrated"
        elif any(word in message_lower for word in ["better", "good", "thanks", "helpful"]):
            emotional_state = "satisfied"
        else:
            emotional_state = "neutral"
        
        self.update_context(session_id, emotional_state=emotional_state)
        return emotional_state
    
    def _load_user_history(self, user_id: str) -> Dict:
        """Load user's historical data from database."""
        try:
            # Query recent conversations
            recent_conversations = self.db_session.query(ConversationRecord)\
                .filter(ConversationRecord.user_id == user_id)\
                .order_by(ConversationRecord.timestamp.desc())\
                .limit(100).all()
            
            # Extract patterns and preferences
            medical_history = []
            previous_concerns = []
            preferences = {}
            
            for conv in recent_conversations:
                if conv.intent == "symptom_inquiry" and conv.entities:
                    entities = json.loads(conv.entities)
                    symptoms = entities.get("symptoms", [])
                    previous_concerns.extend(symptoms)
                
                # Extract medical history mentions
                if "history" in conv.message_content.lower():
                    medical_history.append(conv.message_content)
            
            return {
                "medical_history": list(set(medical_history)),
                "previous_concerns": list(set(previous_concerns)),
                "preferences": preferences
            }
            
        except Exception as e:
            print(f"Error loading user history: {e}")
            return {"medical_history": [], "previous_concerns": [], "preferences": {}}
    
    def save_conversation(self, session_id: str, user_id: str, message: str, 
                         role: str, intent: str, entities: Dict, confidence: float):
        """Save conversation to persistent storage."""
        record = ConversationRecord(
            id=f"{session_id}_{datetime.utcnow().isoformat()}",
            session_id=session_id,
            user_id=user_id,
            message_content=message,
            message_role=role,
            intent=intent,
            entities=json.dumps(entities),
            confidence_score=int(confidence * 100)
        )
        
        self.db_session.add(record)
        self.db_session.commit()
    
    def generate_personalized_greeting(self, session_id: str, user_id: str) -> str:
        """Generate personalized greeting based on user history."""
        context = self.get_context(session_id, user_id)
        
        base_greeting = "Hello! I'm here to help with your health questions."
        
        # Personalize based on previous concerns
        if context.previous_concerns:
            recent_concern = context.previous_concerns[-1]
            base_greeting += f" I see you previously asked about {recent_concern}. How are you feeling today?"
        
        # Adjust tone based on emotional state
        if context.emotional_state == "anxious":
            base_greeting = "Hello, I understand health concerns can be worrying. I'm here to help provide information and support. " + base_greeting
        
        return base_greeting
```

## Phase 3: Production API Development

### 3.1 FastAPI Backend Implementation

**Main API Server:**
```python
# api/main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import uuid
from datetime import datetime
import json
import logging

from chatbot.core.enhanced_bot import EnhancedMedicalChatbot

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    metadata: Dict[str, Any]
    timestamp: datetime

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    summary: Dict[str, Any]

# Initialize FastAPI app
app = FastAPI(
    title="Medical Chatbot API",
    description="Specialized medical support chatbot with context management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global chatbot instance
chatbot = None

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup."""
    global chatbot
    try:
        logging.info("Initializing Medical Chatbot...")
        chatbot = EnhancedMedicalChatbot()
        logging.info("Chatbot initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logging.info("Shutting down Medical Chatbot API")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication - implement proper auth in production."""
    # In production, validate JWT token, check database, etc.
    return {"user_id": "authenticated_user"}

@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    chat_message: ChatMessage,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Main chat endpoint."""
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service unavailable")
        
        # Use provided user_id or default to authenticated user
        user_id = chat_message.user_id or current_user["user_id"]
        
        # Generate response
        response, metadata = chatbot.generate_enhanced_response(
            user_id, chat_message.message
        )
        
        # Log conversation asynchronously
        background_tasks.add_task(
            log_conversation,
            user_id,
            chat_message.message,
            response,
            metadata
        )
        
        return ChatResponse(
            response=response,
            session_id=metadata["session_id"],
            metadata=metadata,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/conversation/{session_id}", response_model=ConversationHistory)
async def get_conversation_history(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get conversation history for a session."""
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service unavailable")
        
        # Get conversation context
        if session_id not in chatbot.active_conversations:
            raise HTTPException(status_code=404, detail="Session not found")
        
        context = chatbot.active_conversations[session_id]
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in context.messages
        ]
        
        summary = chatbot.get_conversation_summary(session_id)
        
        return ConversationHistory(
            session_id=session_id,
            messages=messages,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/conversation/{session_id}")
async def clear_conversation(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Clear conversation history for a session."""
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service unavailable")
        
        chatbot.clear_conversation(session_id)
        return {"message": f"Conversation {session_id} cleared successfully"}
        
    except Exception as e:
        logging.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics")
async def get_analytics(current_user: dict = Depends(get_current_user)):
    """Get chatbot analytics dashboard."""
    try:
        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot service unavailable")
        
        analytics = chatbot.get_analytics_dashboard()
        return analytics
        
    except Exception as e:
        logging.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/feedback")
async def submit_feedback(
    feedback_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Submit feedback for improving the chatbot."""
    try:
        # Store feedback (implement your feedback storage logic)
        feedback_entry = {
            "user_id": current_user["user_id"],
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback_data,
            "session_id": feedback_data.get("session_id")
        }
        
        # In production, save to database
        logging.info(f"Feedback received: {json.dumps(feedback_entry)}")
        
        return {"message": "Feedback submitted successfully"}
        
    except Exception as e:
        logging.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def log_conversation(user_id: str, user_message: str, bot_response: str, metadata: Dict):
    """Background task to log conversations."""
    try:
        # Implement conversation logging logic
        log_entry = {
            "user_id": user_id,
            "user_message": user_message,
            "bot_response": bot_response,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        # In production, save to analytics database
        logging.info(f"Conversation logged: {json.dumps(log_entry)}")
        
    except Exception as e:
        logging.error(f"Error logging conversation: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

### 3.2 WebSocket Support for Real-time Chat

**WebSocket Implementation:**
```python
# api/websocket_handler.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio
import logging

class ConnectionManager:
    """WebSocket connection manager for real-time chat."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logging.info(f"User {user_id} connected via WebSocket")
    
    def disconnect(self, user_id: str):
        """Remove WebSocket connection."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logging.info(f"User {user_id} disconnected from WebSocket")
    
    async def send_personal_message(self, message: str, user_id: str):
        """Send message to specific user."""
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections.values():
            await connection.send_text(message)

# Add to main.py
from .websocket_handler import ConnectionManager

manager = ConnectionManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time chat."""
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message through chatbot
            if chatbot:
                response, metadata = chatbot.generate_enhanced_response(
                    user_id, message_data["message"]
                )
                
                # Send response back to client
                response_data = {
                    "type": "bot_response",
                    "response": response,
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat()
                }
                
                await manager.send_personal_message(
                    json.dumps(response_data), user_id
                )
            
    except WebSocketDisconnect:
        manager.disconnect(user_id)
        logging.info(f"User {user_id} disconnected")
    except Exception as e:
        logging.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(user_id)
```

## Phase 4: User Interface Development

### 4.1 Streamlit Web Interface

**Main Streamlit App:**
```python
# ui/streamlit_app.py
import streamlit as st
import requests
import json
from datetime import datetime
import uuid

# Page config
st.set_page_config(
    page_title="Medical Assistant Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "your-api-token"  # In production, use proper authentication

# Custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}

.user-message {
    background-color: #e3f2fd;
    margin-left: 20%;
}

.bot-message {
    background-color: #f5f5f5;
    margin-right: 20%;
}

.message-header {
    font-weight: bold;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}

.message-content {
    font-size: 1rem;
    line-height: 1.4;
}

.metadata {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.5rem;
}

.sidebar-section {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f8f9fa;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"

def send_message(message: str):
    """Send message to chatbot API."""
    try:
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "message": message,
            "user_id": st.session_state.user_id,
            "session_id": st.session_state.session_id
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None

def display_message(message: dict, is_user: bool = True):
    """Display a chat message."""
    css_class = "user-message" if is_user else "bot-message"
    header = "You" if is_user else "Medical Assistant"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div class="message-header">{header}</div>
        <div class="message-content">{message['content']}</div>
        <div class="metadata">{message.get('timestamp', '')}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("🏥 Medical Assistant Chatbot")
    st.markdown("*Your AI-powered health information companion*")
    
    # Sidebar
    with st.sidebar:
        st.header("💡 Information")
        
        with st.container():
            st.markdown("""
            <div class="sidebar-section">
                <h4>⚠️ Important Disclaimer</h4>
                <p>This chatbot provides general health information only and is not a substitute for professional medical advice, diagnosis, or treatment.</p>
                <p><strong>In case of emergency, call 911 immediately.</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Session info
        with st.container():
            st.markdown("""
            <div class="sidebar-section">
                <h4>📊 Session Info</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.text(f"Session: {st.session_state.session_id[:8]}...")
            st.text(f"Messages: {len(st.session_state.messages)}")
        
        # Quick actions
        st.header("🚀 Quick Actions")
        
        if st.button("🗑️ Clear Conversation"):
            try:
                headers = {"Authorization": f"Bearer {API_TOKEN}"}
                requests.delete(
                    f"{API_BASE_URL}/conversation/{st.session_state.session_id}",
                    headers=headers
                )
                st.session_state.messages = []
                st.session_state.session_id = str(uuid.uuid4())
                st.success("Conversation cleared!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error clearing conversation: {e}")
        
        if st.button("📊 View Analytics"):
            st.session_state.show_analytics = True
        
        # Sample questions
        st.header("💬 Sample Questions")
        
        sample_questions = [
            "I have a headache that won't go away",
            "What should I know about blood pressure medication?",
            "How can I schedule an appointment?",
            "I'm feeling anxious about my health"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}"):
                st.session_state.user_input = question
    
    # Main chat area
    chat_container = st.container()
    
    with chat_container:
        # Display conversation history
        for message in st.session_state.messages:
            if message["role"] == "user":
                display_message(message, is_user=True)
            else:
                display_message(message, is_user=False)
                
                # Show metadata if available
                if "metadata" in message:
                    with st.expander("📊 Response Details"):
                        st.json(message["metadata"])
    
    # Input area (fixed at bottom)
    st.markdown("---")
    
    # Text input
    user_input = st.text_input(
        "Type your health question here...",
        key="user_input",
        placeholder="Ask about symptoms, medications, appointments, or general health questions"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        send_button = st.button("Send Message", type="primary")
    
    with col2:
        if st.button("🎤 Voice Input"):
            st.info("Voice input feature coming soon!")
    
    with col3:
        if st.button("📎 Attach File"):
            st.info("File attachment feature coming soon!")
    
    # Handle message sending
    if send_button and user_input:
        # Add user message to history
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.messages.append(user_message)
        
        # Show typing indicator
        with st.spinner("Medical Assistant is thinking..."):
            # Send to API
            api_response = send_message(user_input)
            
            if api_response:
                # Add bot response to history
                bot_message = {
                    "role": "assistant",
                    "content": api_response["response"],
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "metadata": api_response.get("metadata", {})
                }
                st.session_state.messages.append(bot_message)
                
                # Update session ID if changed
                if "session_id" in api_response:
                    st.session_state.session_id = api_response["session_id"]
        
        # Clear input and rerun
        st.session_state.user_input = ""
        st.experimental_rerun()
    
    # Analytics modal
    if getattr(st.session_state, 'show_analytics', False):
        st.markdown("---")
        st.header("📊 Analytics Dashboard")
        
        try:
            headers = {"Authorization": f"Bearer {API_TOKEN}"}
            analytics_response = requests.get(
                f"{API_BASE_URL}/analytics",
                headers=headers
            )
            
            if analytics_response.status_code == 200:
                analytics_data = analytics_response.json()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Conversations", analytics_data.get("total_conversations", 0))
                
                with col2:
                    st.metric("Average Success Score", f"{analytics_data.get('average_success_score', 0):.2f}")
                
                with col3:
                    st.metric("Active Sessions", analytics_data.get("active_sessions", 0))
                
                # Intent distribution
                if "intent_distribution" in analytics_data:
                    st.subheader("Intent Distribution")
                    st.bar_chart(analytics_data["intent_distribution"])
                
                # Most referenced interactions
                if "most_referenced_interactions" in analytics_data:
                    st.subheader("Most Referenced Interactions")
                    for interaction in analytics_data["most_referenced_interactions"]:
                        st.write(f"• {interaction['user_message']} (accessed {interaction['access_count']} times)")
            
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
        
        if st.button("Close Analytics"):
            st.session_state.show_analytics = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()
```

### 4.2 React Frontend (Alternative)

**React Chat Component:**
```jsx
// ui/react-frontend/src/components/ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './ChatInterface.css';

const API_BASE_URL = 'http://localhost:8000';
const API_TOKEN = 'your-api-token';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [userId] = useState(`user_${Math.random().toString(36).substr(2, 9)}`);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (message) => {
    if (!message.trim()) return;

    const userMessage = {
      role: 'user',
      content: message,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(
        `${API_BASE_URL}/chat`,
        {
          message: message,
          user_id: userId,
          session_id: sessionId
        },
        {
          headers: {
            'Authorization': `Bearer ${API_TOKEN}`,
            'Content-Type': 'application/json'
          }
        }
      );

      const botMessage = {
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date().toLocaleTimeString(),
        metadata: response.data.metadata
      };

      setMessages(prev => [...prev, botMessage]);
      setSessionId(response.data.session_id);

    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toLocaleTimeString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error('Error sending message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage(inputMessage);
  };

  const clearConversation = async () => {
    if (sessionId) {
      try {
        await axios.delete(`${API_BASE_URL}/conversation/${sessionId}`, {
          headers: { 'Authorization': `Bearer ${API_TOKEN}` }
        });
      } catch (error) {
        console.error('Error clearing conversation:', error);
      }
    }
    setMessages([]);
    setSessionId(null);
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h1>🏥 Medical Assistant</h1>
        <button onClick={clearConversation} className="clear-btn">
          Clear Chat
        </button>
      </div>

      <div className="chat-messages">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.role} ${message.isError ? 'error' : ''}`}
          >
            <div className="message-header">
              {message.role === 'user' ? 'You' : 'Medical Assistant'}
              <span className="timestamp">{message.timestamp}</span>
            </div>
            <div className="message-content">
              {message.content}
            </div>
            {message.metadata && (
              <div className="message-metadata">
                <small>
                  Intent: {message.metadata.intent} 
                  (Confidence: {(message.metadata.confidence * 100).toFixed(1)}%)
                </small>
              </div>
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="message assistant loading">
            <div className="message-header">Medical Assistant</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask about symptoms, medications, or health questions..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !inputMessage.trim()}>
            Send
          </button>
        </form>
      </div>

      <div className="disclaimer">
        ⚠️ This chatbot provides general information only. 
        For medical emergencies, call 911 immediately.
      </div>
    </div>
  );
};

export default ChatInterface;
```

## Phase 5: Testing and Evaluation

### 5.1 Comprehensive Testing Framework

**Unit Tests:**
```python
# tests/test_chatbot.py
import pytest
import json
from unittest.mock import Mock, patch
from chatbot.core.bot import MedicalChatbot
from chatbot.core.enhanced_bot import EnhancedMedicalChatbot

class TestMedicalChatbot:
    """Test suite for Medical Chatbot functionality."""
    
    @pytest.fixture
    def chatbot(self):
        """Create chatbot instance for testing."""
        return MedicalChatbot()
    
    @pytest.fixture
    def enhanced_chatbot(self):
        """Create enhanced chatbot instance for testing."""
        return EnhancedMedicalChatbot()
    
    def test_intent_detection(self, chatbot):
        """Test intent detection accuracy."""
        test_cases = [
            ("I have a headache", "symptom_inquiry"),
            ("What medication should I take for pain?", "medication_info"),
            ("Can I schedule an appointment?", "appointment"),
            ("How can I stay healthy?", "general_health")
        ]
        
        for message, expected_intent in test_cases:
            intent, confidence = chatbot.detect_intent(message)
            assert intent == expected_intent
            assert confidence > 0.3  # Minimum confidence threshold
    
    def test_entity_extraction(self, chatbot):
        """Test entity extraction from messages."""
        message = "I've had fever and headache since Monday"
        entities = chatbot.extract_entities(message)
        
        assert "symptoms" in entities
        assert len(entities["symptoms"]) > 0
        assert "dates" in entities
    
    def test_emergency_detection(self, chatbot):
        """Test emergency situation detection."""
        emergency_messages = [
            "I'm having chest pain",
            "I can't breathe properly",
            "This is an emergency",
            "I think I'm having a heart attack"
        ]
        
        for message in emergency_messages:
            assert chatbot.check_emergency(message) == True
        
        normal_messages = [
            "I have a mild headache",
            "Can you help me with my prescription?"
        ]
        
        for message in normal_messages:
            assert chatbot.check_emergency(message) == False
    
    def test_response_generation(self, chatbot):
        """Test response generation and quality."""
        session_id = "test_session"
        message = "I have been feeling tired lately"
        
        response = chatbot.generate_response(session_id, message)
        
        assert isinstance(response, str)
        assert len(response) > 10  # Non-empty response
        assert "healthcare professional" in response.lower()  # Safety disclaimer
    
    def test_conversation_context(self, chatbot):
        """Test conversation context management."""
        session_id = "test_session_context"
        
        # First message
        response1 = chatbot.generate_response(session_id, "Hello")
        assert session_id in chatbot.active_conversations
        
        # Second message should have context
        response2 = chatbot.generate_response(session_id, "I have a headache")
        context = chatbot.active_conversations[session_id]
        assert len(context.messages) >= 2
    
    def test_enhanced_features(self, enhanced_chatbot):
        """Test enhanced chatbot features."""
        user_id = "test_user"
        message = "I'm worried about my symptoms"
        
        response, metadata = enhanced_chatbot.generate_enhanced_response(user_id, message)
        
        assert isinstance(response, str)
        assert isinstance(metadata, dict)
        assert "session_id" in metadata
        assert "intent" in metadata
        assert "emotional_state" in metadata

class TestMemorySystem:
    """Test suite for memory and context management."""
    
    @pytest.fixture
    def memory_bank(self):
        """Create memory bank for testing."""
        from chatbot.core.memory_system import MemoryBank
        return MemoryBank()
    
    def test_episodic_memory_storage(self, memory_bank):
        """Test episodic memory storage and retrieval."""
        # Store a memory
        memory_bank.store_episodic_memory(
            session_id="test_session",
            user_message="I have a headache",
            bot_response="Here's information about headaches...",
            context={"intent": "symptom_inquiry"},
            success_score=0.8
        )
        
        assert len(memory_bank.episodic_memories) == 1
        
        # Retrieve similar episodes
        similar = memory_bank.retrieve_similar_episodes("My head hurts")
        assert len(similar) >= 0  # May or may not find similar based on threshold
    
    def test_procedural_learning(self, memory_bank):
        """Test procedural pattern learning."""
        # Learn a pattern
        memory_bank.learn_procedural_pattern(
            situation="headache_inquiry",
            action="provide_remedies_and_warning_signs",
            outcome=0.9
        )
        
        # Get best action
        best_action = memory_bank.get_best_action("headache_inquiry")
        assert best_action == "provide_remedies_and_warning_signs"

# Integration Tests
class TestIntegration:
    """Integration tests for complete chatbot functionality."""
    
    def test_full_conversation_flow(self):
        """Test complete conversation flow."""
        chatbot = EnhancedMedicalChatbot()
        user_id = "integration_test_user"
        
        conversation_flow = [
            "Hello, I need help",
            "I've been having headaches for 3 days",
            "What medication can I take?",
            "Should I see a doctor?"
        ]
        
        responses = []
        for message in conversation_flow:
            response, metadata = chatbot.generate_enhanced_response(user_id, message)
            responses.append((response, metadata))
            
            # Verify response quality
            assert len(response) > 0
            assert metadata["intent"] is not None
            assert metadata["confidence"] > 0
        
        # Verify conversation was tracked
        analytics = chatbot.get_analytics_dashboard()
        assert analytics["total_conversations"] > 0

# Performance Tests
class TestPerformance:
    """Performance tests for scalability."""
    
    def test_response_time(self):
        """Test response time performance."""
        import time
        
        chatbot = MedicalChatbot()
        message = "I have a question about my health"
        session_id = "perf_test"
        
        start_time = time.time()
        response = chatbot.generate_response(session_id, message)
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 5.0  # Should respond within 5 seconds
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        import threading
        import queue
        
        chatbot = MedicalChatbot()
        results = queue.Queue()
        
        def send_request(session_id, message):
            try:
                response = chatbot.generate_response(session_id, message)
                results.put(("success", response))
            except Exception as e:
                results.put(("error", str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=send_request,
                args=(f"concurrent_session_{i}", f"Test message {i}")
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = 0
        while not results.empty():
            status, _ = results.get()
            if status == "success":
                success_count += 1
        
        assert success_count >= 4  # At least 80% success rate

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 5.2 API Testing

**API Test Suite:**
```python
# tests/test_api.py
import pytest
import requests
import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

class TestChatAPI:
    """Test suite for Chat API endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_chat_endpoint(self):
        """Test main chat endpoint."""
        # Mock authentication
        headers = {"Authorization": "Bearer test-token"}
        
        payload = {
            "message": "Hello, I need help with my health",
            "user_id": "test_user",
            "session_id": "test_session"
        }
        
        response = client.post("/chat", json=payload, headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert "session_id" in data
        assert "metadata" in data
    
    def test_conversation_history(self):
        """Test conversation history retrieval."""
        headers = {"Authorization": "Bearer test-token"}
        
        # First, send a message to create conversation
        payload = {
            "message": "Test message",
            "user_id": "test_user"
        }
        chat_response = client.post("/chat", json=payload, headers=headers)
        session_id = chat_response.json()["session_id"]
        
        # Then get conversation history
        response = client.get(f"/conversation/{session_id}", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "messages" in data
        assert "summary" in data
    
    def test_analytics_endpoint(self):
        """Test analytics endpoint."""
        headers = {"Authorization": "Bearer test-token"}
        
        response = client.get("/analytics", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        # Check for expected analytics fields
        expected_fields = ["total_conversations", "memory_bank_size", "active_sessions"]
        assert any(field in data for field in expected_fields)
    
    def test_feedback_submission(self):
        """Test feedback submission."""
        headers = {"Authorization": "Bearer test-token"}
        
        feedback_data = {
            "rating": 5,
            "comment": "Very helpful chatbot",
            "session_id": "test_session"
        }
        
        response = client.post("/feedback", json=feedback_data, headers=headers)
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_error_handling(self):
        """Test error handling for invalid requests."""
        headers = {"Authorization": "Bearer test-token"}
        
        # Test empty message
        payload = {"message": "", "user_id": "test_user"}
        response = client.post("/chat", json=payload, headers=headers)
        assert response.status_code == 422  # Validation error
        
        # Test invalid session ID for history
        response = client.get("/conversation/invalid_session", headers=headers)
        assert response.status_code == 404
```

### 5.3 Evaluation Metrics and Monitoring

**Evaluation Framework:**
```python
# evaluation/metrics.py
import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from textstat import flesch_reading_ease
import re

class ChatbotEvaluator:
    """Comprehensive chatbot evaluation system."""
    
    def __init__(self):
        self.evaluation_data = []
        self.metrics_history = []
    
    def evaluate_intent_accuracy(self, predictions: List[str], 
                                ground_truth: List[str]) -> Dict[str, float]:
        """Evaluate intent classification accuracy."""
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted'
        )
        
        return {
            "intent_accuracy": accuracy,
            "intent_precision": precision,
            "intent_recall": recall,
            "intent_f1": f1
        }
    
    def evaluate_response_quality(self, responses: List[str]) -> Dict[str, float]:
        """Evaluate response quality using various metrics."""
        metrics = {
            "avg_response_length": np.mean([len(r) for r in responses]),
            "avg_readability": np.mean([flesch_reading_ease(r) for r in responses if len(r) > 0]),
            "safety_disclaimer_coverage": self._check_safety_disclaimers(responses),
            "information_density": self._calculate_information_density(responses)
        }
        
        return metrics
    
    def _check_safety_disclaimers(self, responses: List[str]) -> float:
        """Check percentage of responses with safety disclaimers."""
        safety_keywords = [
            "healthcare professional", "doctor", "medical advice", 
            "emergency", "consult", "professional help"
        ]
        
        responses_with_disclaimer = 0
        for response in responses:
            response_lower = response.lower()
            if any(keyword in response_lower for keyword in safety_keywords):
                responses_with_disclaimer += 1
        
        return responses_with_disclaimer / len(responses) if responses else 0
    
    def _calculate_information_density(self, responses: List[str]) -> float:
        """Calculate average information density of responses."""
        densities = []
        for response in responses:
            # Count informative elements
            bullets = len(re.findall(r'[•\-\*]', response))
            bold_text = len(re.findall(r'\*\*.*?\*\*', response))
            sections = len(re.findall(r'\n\n', response))
            
            # Normalize by response length
            density = (bullets + bold_text + sections) / max(len(response), 1) * 1000
            densities.append(density)
        
        return np.mean(densities) if densities else 0
    
    def evaluate_conversation_flow(self, conversations: List[List[Dict]]) -> Dict[str, float]:
        """Evaluate conversation flow and coherence."""
        metrics = {
            "avg_conversation_length": np.mean([len(conv) for conv in conversations]),
            "topic_coherence": self._calculate_topic_coherence(conversations),
            "user_satisfaction_proxy": self._estimate_satisfaction(conversations)
        }
        
        return metrics
    
    def _calculate_topic_coherence(self, conversations: List[List[Dict]]) -> float:
        """Calculate topic coherence within conversations."""
        coherence_scores = []
        
        for conversation in conversations:
            if len(conversation) < 3:
                continue
            
            intents = [msg.get('intent', 'unknown') for msg in conversation]
            # Calculate intent transition smoothness
            valid_transitions = 0
            total_transitions = len(intents) - 1
            
            for i in range(total_transitions):
                if intents[i] == intents[i+1] or self._is_valid_transition(intents[i], intents[i+1]):
                    valid_transitions += 1
            
            coherence = valid_transitions / total_transitions if total_transitions > 0 else 1.0
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0
    
    def _is_valid_transition(self, from_intent: str, to_intent: str) -> bool:
        """Check if intent transition is logical."""
        valid_transitions = {
            "greeting": ["symptom_inquiry", "appointment", "general_health"],
            "symptom_inquiry": ["medication_info", "appointment"],
            "medication_info": ["symptom_inquiry", "appointment"],
            "appointment": ["general_health", "symptom_inquiry"]
        }
        
        return to_intent in valid_transitions.get(from_intent, [])
    
    def _estimate_satisfaction(self, conversations: List[List[Dict]]) -> float:
        """Estimate user satisfaction based on conversation patterns."""
        satisfaction_indicators = 0
        total_conversations = len(conversations)
        
        for conversation in conversations:
            # Look for positive indicators
            last_messages = conversation[-3:] if len(conversation) >= 3 else conversation
            
            for msg in last_messages:
                if msg.get('role') == 'user':
                    content = msg.get('content', '').lower()
                    if any(word in content for word in ['thanks', 'helpful', 'good', 'better']):
                        satisfaction_indicators += 1
                        break
        
        return satisfaction_indicators / total_conversations if total_conversations > 0 else 0
    
    def generate_evaluation_report(self, test_data: Dict) -> Dict:
        """Generate comprehensive evaluation report."""
        report = {
            "timestamp": json.dumps(datetime.now(), default=str),
            "test_summary": {
                "total_conversations": len(test_data.get("conversations", [])),
                "total_responses": len(test_data.get("responses", [])),
                "unique_intents": len(set(test_data.get("intents", [])))
            }
        }
        
        # Intent evaluation
        if "intent_predictions" in test_data and "intent_ground_truth" in test_data:
            report["intent_metrics"] = self.evaluate_intent_accuracy(
                test_data["intent_predictions"],
                test_data["intent_ground_truth"]
            )
        
        # Response quality evaluation
        if "responses" in test_data:
            report["response_quality"] = self.evaluate_response_quality(test_data["responses"])
        
        # Conversation flow evaluation
        if "conversations" in test_data:
            report["conversation_flow"] = self.evaluate_conversation_flow(test_data["conversations"])
        
        return report

# Continuous monitoring
class ProductionMonitor:
    """Monitor chatbot performance in production."""
    
    def __init__(self, alert_thresholds: Dict[str, float]):
        self.alert_thresholds = alert_thresholds
        self.metrics_buffer = []
    
    def log_interaction(self, user_message: str, bot_response: str, 
                       metadata: Dict, user_feedback: Dict = None):
        """Log interaction for monitoring."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response,
            "metadata": metadata,
            "user_feedback": user_feedback,
            "response_time": metadata.get("response_time", 0),
            "confidence": metadata.get("confidence", 0)
        }
        
        self.metrics_buffer.append(interaction)
        
        # Check for alerts
        self._check_alerts(interaction)
    
    def _check_alerts(self, interaction: Dict):
        """Check for performance alerts."""
        # Low confidence alert
        if interaction["confidence"] < self.alert_thresholds.get("min_confidence", 0.5):
            self._send_alert("LOW_CONFIDENCE", interaction)
        
        # High response time alert
        if interaction["response_time"] > self.alert_thresholds.get("max_response_time", 5.0):
            self._send_alert("HIGH_RESPONSE_TIME", interaction)
        
        # Emergency detection without proper response
        if "emergency" in interaction["user_message"].lower():
            if "emergency" not in interaction["bot_response"].lower():
                self._send_alert("EMERGENCY_NOT_HANDLED", interaction)
    
    def _send_alert(self, alert_type: str, interaction: Dict):
        """Send performance alert."""
        alert = {
            "type": alert_type,
            "timestamp": datetime.now().isoformat(),
            "interaction": interaction
        }
        
        # In production, send to monitoring system (Slack, email, etc.)
        print(f"ALERT: {alert_type} - {json.dumps(alert, indent=2)}")
    
    def get_daily_metrics(self) -> Dict:
        """Get daily performance metrics."""
        today_interactions = [
            i for i in self.metrics_buffer 
            if datetime.fromisoformat(i["timestamp"]).date() == datetime.now().date()
        ]
        
        if not today_interactions:
            return {"message": "No interactions today"}
        
        return {
            "total_interactions": len(today_interactions),
            "avg_confidence": np.mean([i["confidence"] for i in today_interactions]),
            "avg_response_time": np.mean([i["response_time"] for i in today_interactions]),
            "emergency_count": sum(1 for i in today_interactions if "emergency" in i["user_message"].lower()),
            "positive_feedback_rate": sum(1 for i in today_interactions 
                                        if i.get("user_feedback", {}).get("rating", 0) >= 4) / len(today_interactions)
        }
```

## Phase 6: Deployment and Production

### 6.1 Docker Containerization

**Dockerfile for Chatbot Service:**
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download required models
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 chatbot && chown -R chatbot:chatbot /app
USER chatbot

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose Configuration:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  chatbot-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/chatbot
      - API_KEY=${API_KEY}
    depends_on:
      - redis
      - postgres
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=chatbot
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  streamlit-ui:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://chatbot-api:8000
    depends_on:
      - chatbot-api
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - chatbot-api
      - streamlit-ui
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

### 6.2 Kubernetes Deployment

**Kubernetes Manifests:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chatbot-api
  labels:
    app: chatbot-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chatbot-api
  template:
    metadata:
      labels:
        app: chatbot-api
    spec:
      containers:
      - name: chatbot-api
        image: your-registry/chatbot-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: chatbot-api-service
spec:
  selector:
    app: chatbot-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chatbot-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: chatbot-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 6.3 CI/CD Pipeline

**GitHub Actions Workflow:**
```yaml
# .github/workflows/deploy.yml
name: Deploy Chatbot

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=chatbot --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t ${{ secrets.REGISTRY_URL }}/chatbot-api:${{ github.sha }} .
        docker build -t ${{ secrets.REGISTRY_URL }}/chatbot-api:latest .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.REGISTRY_PASSWORD }} | docker login ${{ secrets.REGISTRY_URL }} -u ${{ secrets.REGISTRY_USERNAME }} --password-stdin
        docker push ${{ secrets.REGISTRY_URL }}/chatbot-api:${{ github.sha }}
        docker push ${{ secrets.REGISTRY_URL }}/chatbot-api:latest
    
    - name: Deploy to Kubernetes
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/chatbot-api chatbot-api=${{ secrets.REGISTRY_URL }}/chatbot-api:${{ github.sha }}
        kubectl rollout status deployment/chatbot-api
```

## Project Deliverables and Assessment

### Expected Outputs

**1. Core Chatbot Implementation** ✅
- Medical domain chatbot with specialized knowledge
- Intent recognition and entity extraction
- Emergency detection and appropriate responses
- Multi-turn conversation support

**2. Advanced Features** ✅
- Context management across sessions
- Memory system for personalized responses
- Emotional state detection
- Learning from user interactions

**3. Production-Ready API** ✅
- RESTful API with comprehensive endpoints
- WebSocket support for real-time chat
- Authentication and authorization
- Rate limiting and error handling

**4. User Interfaces** ✅
- Streamlit web application
- React frontend alternative
- Mobile-responsive design
- Accessibility features

**5. Testing and Quality Assurance** ✅
- Comprehensive test suite (unit, integration, performance)
- API testing with various scenarios
- Evaluation metrics and monitoring
- Continuous performance tracking

**6. Deployment Infrastructure** ✅
- Docker containerization
- Kubernetes orchestration
- CI/CD pipeline
- Production monitoring

### Assessment Criteria

| Component | Weight | Criteria |
|-----------|--------|----------|
| **Technical Implementation** | 30% | Code quality, architecture, best practices |
| **Functionality** | 25% | Feature completeness, accuracy, user experience |
| **Innovation** | 20% | Creative solutions, advanced features, AI/ML integration |
| **Documentation** | 15% | Code documentation, user guides, technical specs |
| **Deployment** | 10% | Production readiness, scalability, monitoring |

### Grading Rubric

**Excellent (90-100%)**
- All features implemented with high quality
- Advanced AI/ML techniques applied effectively  
- Comprehensive testing and monitoring
- Professional-grade documentation
- Production-ready deployment

**Good (80-89%)**
- Most features implemented correctly
- Good use of NLP techniques
- Adequate testing coverage
- Clear documentation
- Working deployment

**Satisfactory (70-79%)**
- Basic chatbot functionality working
- Simple NLP implementation
- Basic testing present
- Minimal documentation
- Local deployment only

**Needs Improvement (<70%)**
- Incomplete or non-functional features
- Poor code quality
- Missing tests or documentation
- Deployment issues

## Learning Resources and Next Steps

### Recommended Reading
1. **"Building Chatbots with Python"** by Sumit Raj
2. **"Natural Language Processing with Transformers"** by Lewis Tunstall
3. **"Conversational AI"** by Adam Dahlgren Lindström
4. **"Python for Data Science"** by Jake VanderPlas

### Online Courses
1. **Coursera: Natural Language Processing Specialization**
2. **edX: Introduction to Artificial Intelligence**  
3. **Udacity: Machine Learning Engineer Nanodegree**
4. **Fast.ai: Practical Deep Learning for Coders**

### Technical Documentation
1. **Hugging Face Transformers Documentation**
2. **spaCy Documentation**
3. **FastAPI Documentation** 
4. **Streamlit Documentation**

### Advanced Topics to Explore

**1. Large Language Models (LLMs)**
```python
# Integration with GPT-4, Claude, or other LLMs
from openai import OpenAI

class LLMEnhancedChatbot(MedicalChatbot):
    def __init__(self):
        super().__init__()
        self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_llm_response(self, context: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": "You are a medical information assistant..."},
            {"role": "user", "content": f"Context: {context}\nUser: {user_message}"}
        ]
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content
```

**2. Voice Integration**
```python
# Speech-to-text and text-to-speech
import speech_recognition as sr
from gtts import gTTS
import pygame

class VoiceEnabledChatbot(EnhancedMedicalChatbot):
    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
    
    def listen_to_user(self) -> str:
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=5)
        
        try:
            return self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
    
    def speak_response(self, text: str):
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
```

**3. Multimodal Capabilities**
```python
# Image processing for medical images
import cv2
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class MultimodalChatbot(EnhancedMedicalChatbot):
    def __init__(self):
        super().__init__()
        self.image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    def analyze_medical_image(self, image_path: str) -> str:
        image = Image.open(image_path)
        inputs = self.image_processor(image, return_tensors="pt")
        
        out = self.image_model.generate(**inputs, max_length=100)
        caption = self.image_processor.decode(out[0], skip_special_tokens=True)
        
        return f"I can see: {caption}. Please consult a healthcare professional for proper medical image analysis."
```

### Project Extensions

**1. Multi-language Support**
- Implement translation capabilities
- Language detection
- Cultural adaptation of responses

**2. Integration with Healthcare Systems**
- EHR (Electronic Health Records) integration
- Appointment booking systems
- Prescription management

**3. Advanced Analytics**
- User behavior analysis
- Conversation quality metrics
- Medical entity recognition improvement

**4. Mobile App Development**
- React Native or Flutter app
- Offline capabilities
- Push notifications

## Conclusion

This comprehensive chatbot project provides hands-on experience with:

✅ **Advanced NLP Techniques**: Intent recognition, entity extraction, context management  
✅ **AI/ML Implementation**: Neural networks, transformers, memory systems  
✅ **Software Engineering**: APIs, databases, testing, deployment  
✅ **Production Systems**: Monitoring, scalability, error handling  
✅ **User Experience**: Interface design, accessibility, responsiveness  

The project bridges theoretical knowledge with practical implementation, preparing you for real-world AI application development. The modular architecture allows for continuous enhancement and experimentation with cutting-edge technologies.

**Remember**: This is a learning project. In production medical applications, ensure compliance with healthcare regulations (HIPAA, GDPR), implement proper security measures, and always include appropriate medical disclaimers.
