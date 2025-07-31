from pydantic import BaseModel

class ItemBase(BaseModel):
    name: str
    description: str
    price: int
    is_active: bool = True

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int

    class Config:
        orm_mode = True