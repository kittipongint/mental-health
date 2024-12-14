from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import Dict, List, Optional
from anthropic import Anthropic
import uvicorn
import json
import os
import httpx
import redis
from datetime import datetime
import config  # Import our config module

app = FastAPI()

# Print configuration on startup
config.print_config()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create SQLAlchemy engine
engine = create_engine(
    config.DATABASE_URL,
    pool_pre_ping=True,  # Enable automatic reconnection
    pool_recycle=3600,   # Recycle connections after 1 hour
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis Configuration
redis_client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    password=config.REDIS_PASSWORD,
    decode_responses=True
)

redis_client.ping()

anthropic = Anthropic(api_key=config.ANTHROPIC_API_KEY)

# Database Model
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True)
    role = Column(String(50))
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating database tables: {e}")

# Pydantic Models
class LineChatMessage(BaseModel):
    username: str
    message: str
    replyToken: str

class ChatMessage(BaseModel):
    user_id: int
    message: str

class ChatResponse(BaseModel):
    response: str
    conversation_history: List[Dict]

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Redis functions
def get_redis_key(user_id: str) -> str:
    return f"user_memory:{user_id}"

def get_user_memory(user_id: str) -> List[Dict]:
    try:
        redis_key = get_redis_key(user_id)
        memory_str = redis_client.get(redis_key)
        
        if memory_str is None:
            return []
        
        return json.loads(memory_str)
    except redis.RedisError as e:
        print(f"Redis error: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []

def update_user_memory(user_id: str, role: str, content: str):
    try:
        redis_key = get_redis_key(user_id)
        memory = get_user_memory(user_id)
        
        memory.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        memory = memory[-10:]
        redis_client.set(redis_key, json.dumps(memory), ex=86400)
    except Exception as e:
        print(f"Error updating Redis memory: {e}")

@app.post("/post", response_model=ChatResponse)
def chat_with_claude(chat_message: LineChatMessage, db: Session = Depends(get_db)):
    try:
        llm_chat_message = ChatMessage(user_id=1, message=chat_message.message)
        # TODO: get user id from database

        memory = get_user_memory(llm_chat_message.user_id)
        
        print(chat_message)
        print(llm_chat_message)
        print(memory)

        # context = "\n\n".join([
        #     f"{msg['role']}: {msg['content']}" 
        #     for msg in memory
        # ])
        
        # prompt = (
        #     f"{context}\n\nHuman: {chat_message.message}\n\nAssistant:"
        #     if context
        #     else f"Human: {chat_message.message}\n\nAssistant:"
        # )

        # response = anthropic.messages.create(
        #     model=config.ANTHROPIC_MODEL,
        #     max_tokens=1000,
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #     ]
        # )

        # assistant_response = response.content[0].text

        # Save to database
        db_human_message = Conversation(
            user_id=llm_chat_message.user_id,
            role="human",
            content=llm_chat_message.message
        )
        db_assistant_message = Conversation(
            user_id=llm_chat_message.user_id,
            role="assistant",
            content="xxxx"
        )
        db.add(db_human_message)
        db.add(db_assistant_message)

        print(db_human_message)
        print(db_assistant_message)
        db.commit()

        update_user_memory(llm_chat_message.user_id, "Human", llm_chat_message.message)
        # update_user_memory(chat_message.user_id, "Assistant", assistant_response)

        return ChatResponse(
            # response=assistant_response,
            conversation_history=get_user_memory(llm_chat_message.user_id)
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{user_id}")
def get_conversation(user_id: str, db: Session = Depends(get_db)):
    conversations = db.query(Conversation)\
        .filter(Conversation.user_id == user_id)\
        .order_by(Conversation.timestamp)\
        .all()
    
    return {
        "recent_memory": get_user_memory(user_id),
        "full_history": [
            {
                "role": conv.role,
                "content": conv.content,
                "timestamp": conv.timestamp.isoformat()
            }
            for conv in conversations
        ]
    }

@app.delete("/conversation/{user_id}")
def clear_conversation(user_id: str, db: Session = Depends(get_db)):
    try:
        db.query(Conversation)\
            .filter(Conversation.user_id == user_id)\
            .delete()
        db.commit()
        
        redis_key = get_redis_key(user_id)
        redis_client.delete(redis_key)
        
        return {"message": f"Conversation history cleared for user {user_id}"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoints
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/health/db")
def db_health_check(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        return {"status": "healthy", "message": "Database connection is working"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")

@app.get("/health/redis")
def redis_health_check():
    try:
        redis_client.ping()
        return {"status": "healthy", "message": "Redis connection is working"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis connection failed: {str(e)}")

# # Initialize database on startup
# @app.on_event("startup")
# def startup_event():
#     init_db()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)