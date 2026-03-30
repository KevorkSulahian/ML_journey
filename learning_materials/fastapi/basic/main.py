from typing import List, Optional
from enum import IntEnum
from pydantic import BaseModel, Field
from fastapi import FastAPI

api = FastAPI()

class PriorityLevel(IntEnum):
    LOW = 3
    MEDIUM = 2
    HIGH = 1

class TodoBase(BaseModel):
    todo_description: str = Field(..., max_length=100, description="Description of the todo item")
    todo_name: str = Field(..., max_length=100, description="Name of the todo item")
    priority: PriorityLevel = Field(PriorityLevel.MEDIUM, description="Priority level of the todo item")

class TodoCreate(TodoBase):
    pass

class Todo(TodoBase):
    todo_id: int = Field(..., description="Unique identifier for the todo item") 

class TodoUpdate(BaseModel):
    todo_description: Optional[str] = Field(None, max_length=100, description="Description of the todo item")
    todo_name: Optional[str] = Field(None, max_length=100, description="Name of the todo item")
    priority: Optional[PriorityLevel] = Field(None, description="Priority level of the todo item")

all_todos = [
    Todo(todo_id=1, todo_description="Buy groceries", todo_name="Groceries", priority=PriorityLevel.HIGH),
    Todo(todo_id=2, todo_description="Read a book", todo_name="Reading", priority=PriorityLevel.MEDIUM),
    Todo(todo_id=3, todo_description="Go for a walk", todo_name="Exercise", priority=PriorityLevel.LOW),
    Todo(todo_id=4, todo_description="Write code", todo_name="Coding", priority=PriorityLevel.HIGH),
]

# Get, Post, Put, Delete
@api.get("/")
def index():
    return {"message": "Hello, FastAPI!"}


@api.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int):
    for todo in all_todos:
        if todo.todo_id == todo_id:
            return todo
    return {"error": "Todo not found"}


@api.get("/todos/", response_model=List[Todo])
def get_todos(first_n: Optional[int] = None):
    if first_n is not None:
        return all_todos[:first_n]
    return all_todos


@api.post("/todos/", response_model=Todo)
def create_todo(todo: TodoCreate):
    new_id = max(todo.todo_id for todo in all_todos) + 1

    new_todo = Todo(
        todo_id=new_id,
        todo_description=todo.todo_description,
        todo_name=todo.todo_name,
        priority=todo.priority
    )

    all_todos.append(new_todo)
    return new_todo



@api.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, updated_todo: TodoUpdate):
    for todo in all_todos:
        if todo.todo_id == todo_id:
            if updated_todo.todo_description:
                todo.todo_description = updated_todo.todo_description
            if updated_todo.todo_name:
                todo.todo_name = updated_todo.todo_name
            if updated_todo.priority:
                todo.priority = updated_todo.priority
            return todo
    return {"error": "Todo not found"}

@api.delete("/todos/{todo_id}", response_model=Todo)
def delete_todo(todo_id: int):
    for index, todo in enumerate(all_todos):
        if todo.todo_id == todo_id:
            deleted_todo = all_todos.pop(index)
            return deleted_todo
    return {"error": "Todo not found"}