"""Reusable LangGraph nodes.

This module contains simple functions implementing each step
of the learning module workflow so they can be composed into
LangGraph graphs.
"""
from __future__ import annotations

import json
import os
from typing import Dict

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
        approval = input("\nDoes this plan look good? (y/n): ").strip().lower()
        if approval in ['y', 'yes']:
            state["approved"] = True
            break
        elif approval in ['n', 'no']:
            state["approved"] = False
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")
    
    return state


def generate(state: Dict, llm: ChatOpenAI) -> Dict:
    """Generate notebook cells for the given outline."""
    if not state.get("approved", False):
        print("Plan was not approved. Exiting.")
        state["cells"] = ""
        return state
        
    prompt = f"""Generate Jupyter notebook cells for this lesson outline:

Topic: {state['topic']}
Outline: {state['outline']}

Create a comprehensive learning module with:
1. A markdown introduction cell explaining the topic
2. Code cells with practical examples and exercises
3. Markdown cells with explanations between code sections
4. Comments in code cells to guide learning

Format the output as a valid JSON structure representing notebook cells.
Each cell should have:
- "cell_type": "markdown" or "code"
- "source": array of strings (each line as a separate string)
- "metadata": empty object {{}}

For code cells, also include:
- "execution_count": null
- "outputs": []

Make sure the content is educational, practical, and suitable for learning."""

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
        except json.JSONDecodeError:
            # If parsing fails, create a simple notebook with the content as markdown
            cells = [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# {state['topic']}\n\n"]
                },
                {
                    "cell_type": "markdown", 
                    "metadata": {},
                    "source": [state['outline']]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
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
                    "name": "python",
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
        state["notebook_file"] = None
    
    return state
