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
import re
import demjson3
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

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    firstname = Column(String, unique=False, index=False)
    lastname = Column(String, unique=False, index=False)
    province = Column(String, unique=False, index=False)
    created_at = Column(DateTime, unique=False, index=False, default=datetime.utcnow)
    updated_at = Column(DateTime, unique=False, index=False)

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

class LineChatResponse(BaseModel):
    username: str
    replyMessage: str
    replyToken: str

def parse_loose_json(json_str):
    try:
        return demjson3.decode(json_str)
    except demjson3.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    
def parse_invalid_json(json_str):
    try:
        # Method 1: Basic cleanup
        # Remove newlines and extra spaces within the string values
        cleaned_json = re.sub(r'\s+', ' ', json_str)
        return json.loads(cleaned_json)
    except json.JSONDecodeError:
        try:
            # Method 2: More aggressive cleaning
            # Remove all newlines and normalize spaces
            cleaned_json = json_str.replace('\n', '').replace('\r', '')
            cleaned_json = ' '.join(cleaned_json.split())
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            try:
                # Method 3: Handle newlines in string values
                # Replace newlines with \n escape sequence
                cleaned_json = json_str.replace('\n', '\\n').replace('\r', '\\r')
                cleaned_json = ' '.join(cleaned_json.split())
                return json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                return None

# More comprehensive version
def clean_and_parse_json(json_str):
    try:
        # Step 1: Remove BOM if present
        if json_str.startswith('\ufeff'):
            json_str = json_str[1:]

        # Step 2: Handle newlines and spaces within string values
        in_string = False
        quote_char = None
        cleaned = []
        i = 0
        while i < len(json_str):
            char = json_str[i]

            # Handle quotes
            if char in ['"', "'"] and (i == 0 or json_str[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    quote_char = char
                elif char == quote_char:
                    in_string = False
                    quote_char = None

            # Handle characters
            if in_string:
                if char in ['\n', '\r']:
                    cleaned.append(' ')  # Replace newlines with space in strings
                else:
                    cleaned.append(char)
            else:
                if not char.isspace():  # Skip whitespace outside strings
                    cleaned.append(char)

            i += 1

        # Step 3: Join and parse
        cleaned_json = ''.join(cleaned)
        return json.loads(cleaned_json)

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    
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

def get_or_create_user(username: str, db: Session = Depends(get_db)):
    try:
        # Try to find the user by username
        existing_user = db.query(User).filter(User.username == username).first()
        
        if existing_user:
            # If user exists, return their ID
            return existing_user
            # return {"user_id": existing_user.id, "username": existing_user.username, "created": False, "firstname": existing_user.firstname, "lastname": existing_user.lastname, "province": existing_user.province}
        
        # If user doesn't exist, create new user
        new_user = User(username=username)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return new_user        
        # return {"user_id": new_user.id, "username": new_user.username, "created": True}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        db.close()

@app.post("/post", response_model=LineChatResponse)
def chat_with_claude(chat_message: LineChatMessage, db: Session = Depends(get_db)):
    try:
        user = get_or_create_user(chat_message.username, db)
        llm_chat_message = ChatMessage(user_id=user.id, message=chat_message.message)
        # TODO: get user id from database, create if not exist

        # memory = get_user_memory(llm_chat_message.user_id)
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

        # TODO: determine system prompt base on current conversation - need help, just talk, say sorry, wanna keep best relation

        # case: need help/advice
        response = anthropic.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1000,
            temperature=.4,
            system="You are a mental health counselor and family violence prevention advisor who provides only positive guidance. The responses will be given in a message format for counseling purposes and shortend as possible and in THAI. and include violence actioner in English like father, mother, child, wife, husband, etc. and include a risk assessment score on a scale of 1-10 indicating the likelihood of violence occurring, And give me the detail of score of indicator, and context summary by ignore harmfull word as `context`, all of that provide as parsable json format: {\"message\": \"\",\"context\":\"violenceActor\": \"\",\"riskScore\": _,\"scoreIndicators\": {\"severity\": _,\"frequency\": _,\"recentEscalation\": _,\"weaponUse\": _,\"threatToLife\": _}\n}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": llm_chat_message.message
                        }
                    ]
                }
                # {
                #     "role": "assistant",
                #     "content": [
                #         {
                #             "type": "text",
                #             "text": "ใครทำคุณบ้าง? ความรุนแรงที่ได้รับจากแต่ละคนหนักขนาดไหน วัดจากระดับ 1-10"
                #         }
                #     ]
                # },
                # # TODO: summary based on actioner and risk level
                # {
                #     "role": "user",
                #     "content": [
                #         {
                #             "type": "text",
                #             "text": "ฉันปลอดภัยแล้ว ขอบใจนะ"
                #         }
                #     ]
                # }
            ]
        )
        
        assistant_response = response.content[0].text

        # Save to database
        db_human_message = Conversation(
            user_id=llm_chat_message.user_id,
            role="human",
            content=llm_chat_message.message
        )
        db_assistant_message = Conversation(
            user_id=llm_chat_message.user_id,
            role="assistant",
            content=assistant_response
        )
        db.add(db_human_message)
        db.add(db_assistant_message)
        db.commit()

        update_user_memory(llm_chat_message.user_id, "Human", llm_chat_message.message)
        update_user_memory(llm_chat_message.user_id, "Assistant", assistant_response)

        # raw=ChatResponse(
        #         response=assistant_response,
        #         conversation_history=get_user_memory(llm_chat_message.user_id)
        #     )

        responseObj = clean_and_parse_json(assistant_response)
        # TODO: Save into risks table
        # responseObj['message']
        # responseObj['context']
        # responseObj['violenceActor']
        # responseObj['riskScore']

        return LineChatResponse(
            username=chat_message.username,
            replyMessage=responseObj['message'],
            replyToken=chat_message.replyToken
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{username}")
def get_conversation(username: str, db: Session = Depends(get_db)):    
    user = get_or_create_user(username, db)
    user_id = user.id

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
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)

# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=80,
#         proxy_headers=True,
#         forwarded_allow_ips="*",
#         timeout_keep_alive=65,
#     )