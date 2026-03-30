#!/usr/bin/env python
import sys
from .  crew import AiNewsCrew
from datetime import datetime

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs_array = [
        {
            'topic': 'LLM agents',
            'date': datetime.now().strftime("%Y-%m-%d")
        },
        {
            'topic': 'hugging face',
            'date': datetime.now().strftime("%Y-%m-%d")
        }
    ]
    print("Running AiNewsCrew")
    AiNewsCrew().crew().kickoff_for_each(inputs=inputs_array) 