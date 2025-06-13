"""Reusable LangGraph nodes.

This module contains simple functions implementing each step
of the learning module workflow so they can be composed into
LangGraph graphs.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from langchain_openai import ChatOpenAI


def clarify(state: Dict, llm: ChatOpenAI) -> Dict:
    """Use the provided topic or ask the user for a concise topic clarification."""
    if "topic" not in state or not state["topic"]:
        question = "Please provide a short topic for the learning module"
        response = llm.invoke(question)
        state["topic"] = response.content
    return state


def plan(state: Dict, llm: ChatOpenAI) -> Dict:
    """Create a brief lesson outline for the clarified topic."""
    prompt = f"""Create a high-level outline for a lesson on {state['topic']}. 
    Keep it concise and focused on the main learning objectives and key topics to cover.
    Format it as a clear, structured outline."""
    response = llm.invoke(prompt)
    state["outline"] = response.content
    return state


def ask_approval(state: Dict, llm: ChatOpenAI) -> Dict:
    """Ask user for approval of the plan before continuing."""
    print("\n" + "="*60)
    print("LESSON PLAN")
    print("="*60)
    print(f"Topic: {state['topic']}")
    print("\nOutline:")
    print(state['outline'])
    print("\n" + "="*60)
    
    while True:
        choice = input("\nWhat would you like to do?\n(a) Approve and continue\n(f) Give feedback for changes\n(q) Quit\nChoice (a/f/q): ").strip().lower()
        
        if choice in ['a', 'approve']:
            state["approved"] = True
            state["needs_feedback"] = False
            break
        elif choice in ['f', 'feedback']:
            feedback = input("\nWhat changes would you like to see in the lesson structure?\n> ")
            state["feedback"] = feedback
            state["approved"] = False
            state["needs_feedback"] = True
            break
        elif choice in ['q', 'quit']:
            state["approved"] = False
            state["needs_feedback"] = False
            break
        else:
            print("Please enter 'a' for approve, 'f' for feedback, or 'q' to quit.")
    return state


def revise_plan(state: Dict, llm: ChatOpenAI) -> Dict:
    """Revise the plan based on user feedback."""
    if not state.get("needs_feedback", False) or not state.get("feedback"):
        return state
    
    print("\n🔄 Revising lesson plan based on your feedback...")
    
    prompt = f"""Revise this lesson outline based on the user's feedback:

Original Topic: {state['topic']}
Original Outline: {state['outline']}

User Feedback: {state['feedback']}

Please create a revised outline that addresses the user's feedback while maintaining educational value and structure. Keep it concise and focused on the main learning objectives."""

    response = llm.invoke(prompt)
    state["outline"] = response.content
    
    # Clear feedback flags so we can ask for approval again
    state["feedback"] = ""
    state["needs_feedback"] = False
    
    return state


def generate(state: Dict, llm: ChatOpenAI) -> Dict:
    """Generate notebook cells for the given outline."""
    if not state.get("approved", False):
        print("Plan was not approved. Exiting.")
        state["cells"] = ""
        return state
    
    print("\n📝 Generating notebook content...")
        
    prompt = f"""Generate Jupyter notebook cells for this lesson outline:

Topic: {state['topic']}
Outline: {state['outline']}

Create a hands-on learning module following this pattern:
- CONCEPT (markdown cell explaining a concept)
- PRACTICE (code cell with example + exercise)
- CONCEPT (markdown cell explaining next concept) 
- PRACTICE (code cell with example + exercise)
- Continue alternating...

Guidelines:
- Skip package installation instructions - assume libraries are already installed
- Focus on practical, hands-on learning with immediate code examples
- Each concept should be concise and immediately followed by practice
- Code cells should include both examples AND exercises for the learner to try
- Use clear, educational comments in code
- Make exercises progressively build on previous concepts

Format the output as a valid JSON array of notebook cells. Each cell must have:
- "cell_type": "markdown" or "code"
- "source": array of strings (each line as a separate string)
- "metadata": {{"language": "markdown"}} for markdown cells or {{"language": "python"}} for code cells

For code cells, also include:
- "execution_count": null
- "outputs": []

Return ONLY the JSON array of cells, no additional text or formatting."""

    response = llm.invoke(prompt)
    state["cells"] = response.content
    return state


def save_notebook(state: Dict, llm: ChatOpenAI) -> Dict:
    """Save the generated content as a Jupyter notebook file."""
    if not state.get("approved", False) or not state.get("cells"):
        return state
    
    # Create a safe filename from the topic
    topic = state['topic'].replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    filename = ''.join(c for c in topic if c in safe_chars) + '.ipynb'
    
    try:
        # Try to parse the generated cells as JSON
        cells_content = state['cells']
        if cells_content.startswith('```json'):
            cells_content = cells_content.split('```json')[1].split('```')[0].strip()
        elif cells_content.startswith('```'):
            cells_content = cells_content.split('```')[1].split('```')[0].strip()
        
        try:
            cells = json.loads(cells_content)
            # Ensure each cell has proper metadata with language property
            for cell in cells:
                if "metadata" not in cell:
                    cell["metadata"] = {}
                if "language" not in cell["metadata"]:
                    if cell["cell_type"] == "code":
                        cell["metadata"]["language"] = "python"
                    else:
                        cell["metadata"]["language"] = "markdown"
                        
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing failed: {e}")
            # If parsing fails, create a simple notebook with the content as markdown
            cells = [
                {
                    "cell_type": "markdown",
                    "metadata": {"language": "markdown"},
                    "source": [f"# {state['topic']}\n\n"]
                },
                {
                    "cell_type": "markdown", 
                    "metadata": {"language": "markdown"},
                    "source": [state['outline']]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {"language": "markdown"},
                    "source": ["## Generated Content\n\n", state['cells']]
                }
            ]
    
        # Create the notebook structure
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python", 
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.10.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save the notebook
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        state["notebook_file"] = filename
        print(f"\n✅ Notebook saved as: {filename}")
        
    except Exception as e:
        print(f"\n❌ Error saving notebook: {e}")
        # Print more details for debugging
        print(f"Error details: {str(e)}")
        state["notebook_file"] = None
    
    return state
