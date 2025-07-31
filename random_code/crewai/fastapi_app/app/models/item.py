from sqlalchemy import Column, Integer, String
from app.database.database import Base

class Item(Base):
    __tablename__ = 'items'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    price = Column(Integer)
    is_active = Column(Boolean, default=True)